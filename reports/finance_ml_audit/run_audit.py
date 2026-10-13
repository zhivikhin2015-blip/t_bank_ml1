"""Reproduce trusted local research notebooks without changing source artifacts.
Run each mode in a fresh process. Dependencies: research requirements + PyYAML for skill validation.
"""
import argparse,ast,contextlib,datetime,functools,hashlib,io,json,os,pathlib,platform,resource,shutil,subprocess,time,traceback,zipfile

NAMES={'manual':'manual_baseline_reviewed.ipynb','automatic':'02_automatic_features_reviewed.ipynb','final':'03_final_comparison.ipynb'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
def prepare(source,workspace,mode):
    root=workspace/mode
    if root.exists():raise FileExistsError(f'Use a fresh workspace; refusing overwrite: {root}')
    root.mkdir(parents=True)
    for folder in ['outputs','archive/source_inputs']:shutil.copytree(source/folder,root/folder)
    (root/'work/data/FAR-Trans').mkdir(parents=True)
    csv=source/'work/data/FAR-Trans/transactions.csv'
    if csv.exists():shutil.copy2(csv,root/'work/data/FAR-Trans/transactions.csv')
    else:
        with zipfile.ZipFile(source/'FAR-Trans.zip') as z:raw=z.read('FAR-Trans/transactions.csv')
        assert hashlib.sha256(raw).hexdigest()=='358871fa1311a802799df86e56c3d633513566965b01412d058e0f8fbe6cd54b'
        (root/'work/data/FAR-Trans/transactions.csv').write_bytes(raw)
    return root

def differences(a,b,path=''):
    import numpy as np
    out=[]
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():out.append({'key':path,'kind':'keys'})
        for k in a.keys()&b.keys():out+=differences(a[k],b[k],path+'.'+k)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):out.append({'key':path,'kind':'length'})
        for i,(x,y) in enumerate(zip(a,b)):out+=differences(x,y,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
        if not np.isclose(a,b,rtol=0,atol=1e-12):out.append({'key':path,'kind':'numeric','original':a,'reproduced':b})
    elif a!=b:out.append({'key':path,'kind':'value','original':a,'reproduced':b})
    return out

def run(source,root,mode,out):
    import numpy as np,pandas as pd,sklearn,joblib,scipy
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from IPython.core.formatters import DisplayFormatter
    from threadpoolctl import threadpool_limits
    all_files={str(p.relative_to(source)):digest(p) for folder in ['outputs','work/data/FAR-Trans','archive/source_inputs'] for p in (source/folder).rglob('*') if p.is_file()}
    original_notebook=source/'outputs'/NAMES[mode]
    nb=json.loads(original_notebook.read_text())
    if mode=='final':
        protocol=json.loads((out/'preregistered_protocol.json').read_text())
        assert protocol['evaluation_policy']=='approved_complete_months' and not protocol['test_evaluated']
        assert protocol['source_final_notebook_sha256']==digest(original_notebook)
        assert protocol['evaluation_dates']==[f'2022-{m:02d}-01' for m in range(7,11)]
        for name,file in {'manual_results':'manual_baseline_reviewed_results.json','automatic_results':'automatic_features_reviewed_results.json','manual_model':'manual_baseline_reviewed_model.joblib','automatic_model':'automatic_features_reviewed_model.joblib','manual_control':'manual_control_100_7_reviewed_model.joblib'}.items():
            assert digest(source/'outputs'/file)==protocol['source_artifact_sha256'][name]
        assert digest(root/'work/data/FAR-Trans/transactions.csv')==protocol['transactions_sha256']
        text=''.join(nb['cells'][4]['source']);assert "evaluation_policy='pending'" in text
        nb['cells'][4]['source']=text.replace("evaluation_policy='pending'","evaluation_policy='approved_complete_months'",1).splitlines(keepends=True)
        shutil.copy2(out/'preregistered_protocol.json',root/'outputs/final_comparison_protocol.json')
    path=root/'outputs'/NAMES[mode]
    ns={'__name__':'__main__'};formatter=DisplayFormatter();traces=[];current=[None]
    original_fit=HGB.fit;original_predict=HGB.predict_proba
    @functools.wraps(original_fit)
    def traced_fit(self,X,y,*args,**kwargs):
        if mode=='final':raise AssertionError('Training is prohibited in final evaluation')
        t=time.perf_counter();result=original_fit(self,X,y,*args,**kwargs)
        traces.append({'operation':'fit','cell_index':current[0],'rows':len(X),'features':X.shape[1],'max_iter':self.max_iter,'max_leaf_nodes':self.max_leaf_nodes,'seconds':time.perf_counter()-t})
        return result
    @functools.wraps(original_predict)
    def traced_predict(self,X,*args,**kwargs):
        t=time.perf_counter();result=original_predict(self,X,*args,**kwargs)
        traces.append({'operation':'predict_proba','cell_index':current[0],'rows':len(X),'features':X.shape[1],'max_iter':self.max_iter,'seconds':time.perf_counter()-t})
        return result
    HGB.fit=traced_fit;HGB.predict_proba=traced_predict
    start=time.perf_counter();cells=[];oldcwd=pathlib.Path.cwd();os.chdir(root)
    try:
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            current[0]=i;text=''.join(c['source']);tree=ast.parse(text);stdout=io.StringIO();stderr=io.StringIO();value=None;t=time.perf_counter()
            if mode=='final':assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['fit','partial_fit'] for n in ast.walk(tree))
            try:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    if tree.body and isinstance(tree.body[-1],ast.Expr):
                        exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),f'<{mode}:cell-{i}>','exec'),ns)
                        value=eval(compile(ast.Expression(tree.body[-1].value),f'<{mode}:cell-{i}>','eval'),ns)
                    else:exec(compile(tree,f'<{mode}:cell-{i}>','exec'),ns)
            except Exception:
                (out/f'{mode}_failure.txt').write_text(stdout.getvalue()+stderr.getvalue()+traceback.format_exc());raise
            elapsed=time.perf_counter()-t;count=len(cells)+1;c['execution_count']=count;c['outputs']=[]
            for stream,s in [('stdout',stdout),('stderr',stderr)]:
                if s.getvalue():c['outputs'].append({'output_type':'stream','name':stream,'text':s.getvalue()})
            if value is not None:
                bundle,metadata=formatter.format(value);c['outputs'].append({'output_type':'execute_result','execution_count':count,'data':bundle,'metadata':metadata})
            path.write_text(json.dumps(nb,ensure_ascii=False,indent=1));cells.append({'cell_index':i,'seconds':elapsed})
            print(f'{mode} cell {i}: {elapsed:.2f}s; total {time.perf_counter()-start:.1f}s',flush=True)
    finally:
        os.chdir(oldcwd);HGB.fit=original_fit;HGB.predict_proba=original_predict
    total=time.perf_counter()-start
    record_name={'manual':'manual_baseline_reviewed_results.json','automatic':'automatic_features_reviewed_results.json','final':'final_comparison_results.json'}[mode]
    record=json.loads((root/'outputs'/record_name).read_text());checks=ns['checks'].copy()
    evidence={'mode':mode,'fresh_process':True,'source_notebook_sha256':digest(original_notebook),'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__},'platform':platform.platform(),'cpu':platform.machine(),'thread_limit':4,'single_run_wall_seconds':total,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cells':cells,'model_operations':traces,'checks':checks}
    if mode in ['manual','automatic']:
        old=json.loads((source/'outputs'/record_name).read_text());delta=differences(old,record)
        path_only=lambda x:x['kind']=='value' and any(x['key'].startswith(p) for p in ['.output_directory','.source_artifact_paths.','.manual_control_artifact.path'])
        unexpected=[x for x in delta if not path_only(x)]
        evidence['record_differences']=delta;evidence['unexpected_differences']=unexpected
        assert not unexpected,unexpected
        kind=mode+'_baseline' if mode=='manual' else 'automatic_features'
        frozen=joblib.load(source/'outputs'/('manual_baseline_reviewed_model.joblib' if mode=='manual' else 'automatic_features_reviewed_model.joblib'))
        model=ns['manual_model'] if mode=='manual' else ns['automatic_model'];columns=record['features'];validation=ns['validation']
        with threadpool_limits(limits=4):
            a=frozen['model'].predict_proba(validation[columns])[:,1];b=model.predict_proba(validation[columns])[:,1]
        evidence['validation_probabilities']={'rows':len(a),'max_abs_difference':float(np.max(np.abs(a-b))),'byte_identical':bool(np.array_equal(a,b)),'source_sha256':hashlib.sha256(a.astype('<f8').tobytes()).hexdigest(),'reproduced_sha256':hashlib.sha256(b.astype('<f8').tobytes()).hexdigest()}
        assert np.allclose(a,b,rtol=0,atol=1e-12)
        if mode=='manual':
            cached=joblib.load(source/'outputs/manual_control_100_7_reviewed_model.joblib')
            with threadpool_limits(limits=4):
                x=cached['model'].predict_proba(validation[columns])[:,1];y=ns['trained_models'][(100,7)].predict_proba(validation[columns])[:,1]
            evidence['control_max_abs_difference']=float(np.max(np.abs(x-y)));assert np.allclose(x,y,rtol=0,atol=1e-12)
    else:
        assert record['test_evaluated'] and record['new_training_fits']==0
        assert record['protocol']['evaluation_dates']==[f'2022-{m:02d}-01' for m in range(7,11)]
        evidence['independent_checks']=independent_final_checks(ns)
        nb['metadata']['audit']={'period':'2022-07 through 2022-10','source_notebook_sha256':digest(original_notebook),'new_training_fits':0,'protocol_sha256':digest(out/'preregistered_protocol.json')}
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
        shutil.copy2(path,out/'03_final_comparison_evaluated.ipynb')
        shutil.copy2(root/'outputs/final_comparison_protocol.json',out/'final_comparison_protocol.json')
    assert all(digest(source/p)==h for p,h in all_files.items())
    evidence['source_artifacts_unchanged']=True
    write_json(out/f'{mode}_execution.json',evidence);shutil.copy2(root/'outputs'/record_name,out/f'{mode}_results.json')
    print(f'{mode}: PASS; {sum(t["operation"]=="fit" for t in traces)} fits; {total:.2f}s',flush=True)

def independent_final_checks(ns):
    """Independent explicit metric oracle; does not call notebook metric helpers."""
    import numpy as np,pandas as pd,math
    from sklearn.metrics import log_loss,roc_auc_score
    data=ns['test'];metrics=['Recall@5','Precision@5','NDCG@5','Precision@5_all'];independent={};maxerr=0
    assert len(data)==100000
    for method,top in ns['top_lists'].items():
        rows=[];hit_lookup={key:g for key,g in top.groupby(['customerID','prediction_date'])}
        for key,g in data.groupby(['customerID','prediction_date']):
            relevant=set(g.loc[g.label==1,'ISIN']);t=hit_lookup[key];ordered=t.sort_values('_rank');hits=sum(a in relevant for a in ordered.ISIN)
            dcg=sum((a in relevant)/math.log2(r+1) for r,a in enumerate(ordered.ISIN,1));ideal=sum(1/math.log2(r+1) for r in range(1,min(5,len(relevant))+1))
            rows.append({'customerID':key[0],'prediction_date':key[1],'Recall@5':hits/len(relevant) if relevant else np.nan,'Precision@5':hits/5 if relevant else np.nan,'NDCG@5':dcg/ideal if relevant else np.nan,'Precision@5_all':hits/5})
        frame=pd.DataFrame(rows);independent[method]=frame
        observed=ns['overall'].set_index('method').loc[method]
        for metric in metrics:
            e=abs(frame[metric].mean()-observed[metric]);maxerr=max(maxerr,e);assert e<1e-12,(method,metric,e)
        probs=ns['method_specs'][method].get('probabilities')
        if probs is not None:
            assert abs(log_loss(data.label,probs)-observed['LogLoss'])<1e-12
            assert abs(roc_auc_score(data.label,probs)-observed['ROC-AUC'])<1e-12
    rng=np.random.default_rng(20261013);clients=ns['client_ids'];results=[]
    for _ in range(10):
        draw=rng.integers(0,len(clients),size=len(clients));picked=[clients[i] for i in draw];multiplicity=pd.Series(picked).value_counts()
        auto=independent['automatic_selected'];weights=auto.customerID.map(multiplicity).fillna(0).to_numpy();eligible=auto['Recall@5'].notna().to_numpy();den=weights[eligible].sum()
        for name in ['manual_selected','pair_then_popularity','manual_100_7_control']:
            delta=auto['Recall@5'].to_numpy()-independent[name]['Recall@5'].to_numpy()
            explicit=float(np.sum(delta[eligible]*weights[eligible])/den)
            selected=ns['recall_cases'](data,**ns['method_specs']['automatic_selected']);reference=ns['recall_cases'](data,**ns['method_specs'][name]);d=(selected-reference).dropna();parts=[d[d.index.get_level_values('customerID')==c].to_numpy() for c in picked];expected=float(np.concatenate(parts).mean())
            assert abs(explicit-expected)<1e-12;results.append(explicit)
    # New random example, independent of built-in fixtures: ties, two positives, inactive case.
    date=pd.Timestamp('2022-07-01');fixture=pd.DataFrame({'customerID':['X']*6+['Y']*6,'prediction_date':[date]*12,'ISIN':list('abcdef')*2,'label':[1,1,0,0,0,0]+[0]*6});p=np.tile([.9,.8,.7,.6,.5,.4],2)
    v=ns['evaluate'](fixture,p);assert np.isclose(v['Recall@5'],1) and np.isclose(v['Precision@5'],.4) and np.isclose(v['Precision@5_all'],.2) and np.isclose(v['NDCG@5'],1)
    tied=np.ones(12)*.5;keys=['customerID','prediction_date','ISIN','_rank'];a=ns['top5'](fixture,tied)[keys].reset_index(drop=True);b=ns['top5'](fixture.sample(frac=1,random_state=91),pd.Series(tied,index=fixture.index))[keys].reset_index(drop=True);pd.testing.assert_frame_equal(a,b)
    history=ns['all_buys'];cutoff=pd.Timestamp('2022-07-01');pairs=ns['candidate_pairs'].iloc[:100]
    before=ns['build_month'](history[history.timestamp<cutoff],pairs,cutoff);after=ns['build_month'](history,pairs,cutoff)
    pd.testing.assert_frame_equal(before[ns['all_features']],after[ns['all_features']])
    return {'explicit_metric_oracle_all_methods':True,'max_ranking_metric_error':maxerr,'logloss_auc_library_oracle':True,'bootstrap_first_10_draws_all_comparisons':True,'independent_two_positive_fixture':True,'equal_score_order_invariance':True,'actual_future_events_feature_invariance':True,'rows':len(data),'november_excluded':bool(data.prediction_date.max()<pd.Timestamp('2022-11-01'))}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=pathlib.Path);p.add_argument('--workspace',required=True,type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path);p.add_argument('--mode',required=True,choices=NAMES);p.add_argument('--prepared',action='store_true',help='Use an already isolated workspace created for this audit');args=p.parse_args();args.source=args.source.resolve();args.workspace=args.workspace.resolve();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    root=args.workspace/args.mode if args.prepared else prepare(args.source,args.workspace,args.mode)
    run(args.source,root,args.mode,args.out)
