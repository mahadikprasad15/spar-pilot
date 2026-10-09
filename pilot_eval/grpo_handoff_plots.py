"""Standalone scientific figures for the verified paired-arm report."""
import numpy as np


def plot_handoff(result,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    outputs=[]
    colors={'sft':'#487ca5','grpo':'#b67861','random':'#7e8791'}
    def finish(fig,name,title):
        fig.suptitle(title,fontsize=13)
        fig.tight_layout(rect=(0,0,1,.96))
        for ax in fig.axes:
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.16)
        fig.savefig(directory/name,bbox_inches='tight');plt.close(fig);outputs.append(name)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for arm in ['sft','grpo']:
        if arm=='sft':rows=result['behaviour']['sft']['trajectory']
        else:rows=[dict(step=int(k.split('-')[1]),**v) for k,v in result['behaviour']['grpo']['summaries'].items() if k.startswith('greedy-')]
        x=[r['step'] for r in rows]
        for ax,metric in zip(axes[0],['flexible_accuracy','mean_tokens']):
            ax.plot(x,[r[metric] for r in rows],marker='o',label=arm.upper(),color=colors[arm])
        changes=(rows if arm=='sft' else [r for r in result['behaviour']['grpo']['changes'] if r['mode']=='greedy'])
        for ax,metric in zip(axes[1],['flexible_change','mean_token_change']):
            values=[r[metric]['estimate'] for r in changes]
            bounds=np.array([r[metric]['interval_95'] for r in changes])
            steps=[r['step'] for r in changes]
            ax.vlines(steps,bounds[:,0],bounds[:,1],color=colors[arm],alpha=.8)
            ax.plot(steps,values,marker='o',label=arm.upper(),color=colors[arm])
    for row in axes:
        for ax in row:ax.set_xlabel('Optimizer step');ax.legend()
    axes[0,0].set_ylabel('Greedy flexible-v3 accuracy');axes[0,0].set_ylim(0,1)
    axes[0,1].set_ylabel('Mean generated tokens')
    axes[1,0].set_ylabel('Accuracy change vs checkpoint 0');axes[1,1].set_ylabel('Token change vs checkpoint 0')
    for ax in axes[1]:ax.axhline(0,color='#999999',lw=.8)
    finish(fig,'behaviour.svg','Greedy behaviour and paired changes · 150 identical held-out items')
    # GRPO endpoint sampling is retained separately; SFT sampled data are absent.
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,metric in zip(axes,['flexible_accuracy','mean_tokens']):
        rows=[result['behaviour']['grpo']['summaries'][f'sampled-{step}'] for step in [0,64]]
        ax.plot([0,64],[r[metric] for r in rows],marker='o',color=colors['grpo'])
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Mean correctness (not pass@8)' if metric=='flexible_accuracy' else 'Mean generated tokens')
    axes[0].set_ylim(0,1)
    finish(fig,'sampled-endpoints.svg','GRPO sampled endpoints · 8 observed draws per held-out item')
    views=['question','solution','user'];weights=['token','example']
    for kind in ['write-depth','direction','kl']:
        fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
        for wi,weight in enumerate(weights):
            for vi,view in enumerate(views):
                ax=axes[wi,vi];ax.set_title(f'{view} · {weight} weighting')
                if kind=='write-depth':
                    for arm in ['sft','grpo','random']:
                        for step in ([64] if arm=='random' else [8,16,32,64]):
                            rows=[r for r in result['writes']['magnitudes'][arm] if r['kind']=='block' and r['step']==step and r['view']==view and r['weighting']==weight]
                            rows.sort(key=lambda r:r['layer'])
                            ax.plot([r['layer'] for r in rows],[np.nan if r['relative_write'] is None else r['relative_write'] for r in rows],
                                color=colors[arm],alpha=.4+step/110,linestyle='--' if arm=='random' else '-',label=f'{arm} {step}')
                    ax.set_xlabel('Decoder block');ax.set_ylabel('‖mean Δh‖ / mean ‖h base‖')
                elif kind=='direction':
                    for step in [8,16,32,64]:
                        rows=[r for r in result['writes']['learned_direction_trajectory'] if r['step']==step and r['view']==view and r['weighting']==weight]
                        ax.plot([r['layer'] for r in rows],[np.nan if r['cosine'] is None else r['cosine'] for r in rows],label=f'step {step}')
                    ax.set_ylim(-1.05,1.05);ax.axhline(0,color='#999999',lw=.8)
                    ax.set_xlabel('Decoder block');ax.set_ylabel('Signed SFT–GRPO mean-write cosine')
                else:
                    for arm in ['sft','grpo']:
                        rows=[r for r in result['kl']['kl_trajectory'] if r['arm']==arm and r['view']==view];rows.sort(key=lambda r:r['step'])
                        field=weight+'_mean'
                        ax.plot([r['step'] for r in rows],[np.nan if r[field] is None else r[field] for r in rows],marker='o',label=arm,color=colors[arm])
                    ax.set_xlabel('Optimizer step');ax.set_ylabel('Fixed-context KL (nats)')
                ax.legend(fontsize=7)
        finish(fig,kind+'.svg',{'write-depth':'Relative block-output writes · fixed supplied tokens','direction':'Learned direction trajectory · unresolved directions shown as gaps','kl':'KL(tuned || untuned) · fixed next-target contexts'}[kind])
    # Seven projection facets, separate files by view and weighting.
    from pilot_eval.activation_prepare import PROJECTIONS
    for view in views:
        for weighting in weights:
            fig,axes=plt.subplots(2,4,figsize=(14,7))
            for projection,ax in zip(PROJECTIONS,axes.flat):
                for arm in ['sft','grpo','random']:
                    rows=[r for r in result['writes']['magnitudes'][arm] if r['kind']=='module' and r['step']==64 and r['view']==view and r['weighting']==weighting and r['projection']==projection]
                    rows.sort(key=lambda r:r['layer'])
                    ax.plot([r['layer'] for r in rows],[np.nan if r['relative_write'] is None else r['relative_write'] for r in rows],label=arm,color=colors[arm])
                ax.set_title(projection);ax.set_xlabel('Decoder block');ax.set_ylabel('Mean branch norm / mean base-output norm');ax.legend(fontsize=7)
            axes.flat[-1].axis('off')
            finish(fig,f'modules-{view}-{weighting}.svg',f'Direct-module contribution · step 64 · {view} · {weighting} weighting')
    return outputs
