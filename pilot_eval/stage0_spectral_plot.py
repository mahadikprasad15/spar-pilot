"""Projection figure from saved rows; no model imports or unreported filtering."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_overlap(rows, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    primary={}
    for row in rows:
        if row['step']==64 and row['energy_level']==50 and row['arm'] in ['sft','grpo']:
            primary.setdefault(row['module_id'],{})[row['arm']]=row
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    colors={'q_proj':'#517b9d','k_proj':'#7fa6b4','v_proj':'#789f8d','o_proj':'#a4a77f',
            'gate_proj':'#b49388','up_proj':'#a18ca5','down_proj':'#aa885f'}
    for axis,field,label in zip(axes,['pi_out','pi_in'],['Output factor b','Input factor a']):
        seen=set()
        for module,pair in primary.items():
            if set(pair)!={'sft','grpo'}:continue
            x,y=pair['grpo'][field],pair['sft'][field]
            if x is None or y is None:continue
            kind=module.split('.')[-1]
            axis.scatter(x,y,color=colors[kind],alpha=.75,s=24,
                         label=kind if kind not in seen else None)
            seen.add(kind)
        axis.plot([0,1],[0,1],color='#9ca3a7',linewidth=1,linestyle='--')
        axis.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='GRPO principal projection energy',
                 ylabel='SFT principal projection energy',title=label)
        axis.spines[['top','right']].set_visible(False)
        axis.grid(alpha=.12)
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Stage 0 · checkpoint 64 · base k50 subspaces',fontsize=12)
    fig.savefig(path,format='svg',metadata={'Description':'All finite primary endpoint module projections; diagonal is equal overlap.'})
    fig.savefig(path.with_suffix('.png'),dpi=150)
    plt.close(fig)
