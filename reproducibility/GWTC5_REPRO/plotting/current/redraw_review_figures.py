"""Presentation-only vector redraws from preserved tables and empirical samples."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import h5py
from scipy.stats import wasserstein_distance

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'review_vector_figures'
OUT.mkdir(exist_ok=True)
inputs = {}
outputs = []

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def bind(p):
    inputs[str(p.relative_to(ROOT))] = sha(p)
    return p

def rows(p):
    with bind(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def values(rs, name):
    return np.array([float(r[name]) for r in rs])

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
    'axes.labelsize':9, 'axes.titlesize':10, 'xtick.labelsize':8.5,
    'ytick.labelsize':8.5, 'legend.fontsize':8.5, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'path.simplify':False, 'axes.spines.top':False,
    'axes.spines.right':False})

def clean(ax):
    ax.grid(color='0.88', linewidth=.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))

def save(fig, stem, data):
    fig.canvas.draw()
    for suffix in ('pdf','svg','png'):
        p = OUT / (stem+'.'+suffix)
        fig.savefig(p, dpi=450, facecolor='white')
        outputs.append({'path':p.name, 'sha256':sha(p)})
    plt.close(fig)
    p = OUT / (stem+'_plotted_values.json')
    p.write_text(json.dumps(data, indent=2)+'\n')
    outputs.append({'path':p.name,'sha256':sha(p)})

def stage(r):
    s=r.get('prior_stage',r.get('stage',''))
    if 'confirmatory' in s: return 'Confirmation', 'o', '0.12'
    if 'replication' in s: return 'Replication', '^', '0.42'
    return 'Other primary events', 's', '0.65'

def scatter(ax, rs, x, y):
    for label,marker,color in [('Confirmation','o','0.12'),('Replication','^','0.42'),('Other primary events','s','0.65')]:
        ids=[i for i,r in enumerate(rs) if stage(r)[0]==label]
        if ids: ax.scatter(x[ids],y[ids],s=19,marker=marker,facecolors=color if marker!='s' else 'white',edgecolors=color,linewidths=.65,label=label,zorder=3)
    clean(ax)

def note(ax, text):
    # Keep statistics outside the data rectangle so no event is obscured.
    ax.set_title(ax.get_title()+'\n'+text.replace('\n', '; '), fontsize=8.5)

catalog=ROOT/'work/GWTC5_full_catalog_generalization/results'
events=[r for r in rows(catalog/'generalization_full/full_catalog_event_endpoints.csv') if r['population_A_nondevelopment'].lower()=='true']
assert len(events)==78
tests=rows(catalog/'generalization_full/full_catalog_generalization_tests.csv')
tests={r['endpoint']:r for r in tests if r['population']=='A_nondevelopment'}
fig,axs=plt.subplots(2,2,figsize=(6.5,5.15),layout='constrained')
x=values(events,'network_matched_filter_snr'); y=values(events,'event_median_NW1')
scatter(axs[0,0],events,x,y)
axs[0,0].set(xlabel='Catalog network SNR',ylabel='Median four-parameter NW1',title='(a) SNR association')
note(axs[0,0],rf'$\rho={float(tests["G1"]["effect"]):.3f}$'+'\n'+r'Holm $p=2.00\times10^{-5}$')
x=values(events,'luminosity_distance_nw1'); y=values(events,'chi_eff_nw1')
scatter(axs[0,1],events,x,y)
lim=max(max(x),max(y))*1.08
axs[0,1].plot([0,lim],[0,lim],color='.55',ls='--',lw=.8)
axs[0,1].set(xlim=(0,lim),ylim=(0,lim),xlabel='Luminosity-distance NW1',ylabel=r'Effective-spin NW1',title='(b) Parameter ordering')
note(axs[0,1],r'$\Delta>0$: 75/78'+'\n'+r'Median $\Delta=0.0525$')
scatter(axs[1,0],events,values(events,'maximum_screening_shift'),values(events,'screening_parameter_max_NW1'))
axs[1,0].set(xlabel='Catalog screening score',ylabel='Maximum screening-parameter NW1',title='(c) Screening validation')
note(axs[1,0],rf'$\rho={float(tests["G3"]["effect"]):.3f}$'+'\n'+r'Holm $p=2.00\times10^{-5}$')
d=np.sort(values(events,'delta_nw1')); ranks=np.arange(1,79)
axs[1,1].vlines(ranks,0,d,color='.4',lw=.6)
axs[1,1].scatter(ranks,d,s=7,color='.1'); axs[1,1].axhline(0,color='.2',lw=.8)
axs[1,1].set(xlabel='Events ranked by paired difference',ylabel=r'NW1($\chi_{\rm eff}$) $-$ NW1($d_L$)',title='(d) Event-level exceptions')
clean(axs[1,1])
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
save(fig,'fig_full_catalog_generalization',events)

mech=[r for r in rows(catalog/'mechanism_full/full_catalog_mechanism_event_metrics.csv') if r['population_A_nondevelopment'].lower()=='true']
assert len(mech)==78 and {r['event'] for r in mech}=={r['event'] for r in events}
fig,axs=plt.subplots(3,1,figsize=(6.5,7.2),layout='constrained')
scatter(axs[0],mech,values(mech,'chi_eff_nw1'),values(mech,'joint_q_chi_eff_sliced_w1'))
axs[0].set(xlabel=r'Effective-spin NW1',ylabel='Joint sliced W1',title=r'(a) Joint $(q,\chi_{\rm eff})$ disagreement')
for r in mech:
    _,m,c=stage(r)
    axs[1].plot([0,1],[float(r['chi_p_nw1']),float(r['chi_eff_nw1'])],color=c,marker=m,ms=3,lw=.6,alpha=.8)
clean(axs[1]); axs[1].set_xticks([0,1],[r'$\chi_p$',r'$\chi_{\rm eff}$'])
axs[1].set(xlim=(-.2,1.2),ylabel='NW1',title='(b) Effective spin and precession spin')
scatter(axs[2],mech,values(mech,'chi_eff_share_of_l1_shift'),values(mech,'absolute_shift_pc1_alignment'))
axs[2].set(xlabel=r'$\chi_{\rm eff}$ share of standardized $L_1$ shift',ylabel='Absolute PC1 alignment',title='(c) Geometry of the joint shift',xlim=(-.02,1.02),ylim=(-.02,1.02))
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
save(fig,'fig_full_catalog_mechanism',mech)

joint=rows(ROOT/'work/gwtc5_joint_mechanism_exploratory/joint_geometry_event_metrics.csv')
assert len(joint)==27
fig,axs=plt.subplots(1,2,figsize=(6.5,3.0),layout='constrained')
scatter(axs[0],joint,values(joint,'chi_eff_nw1'),values(joint,'joint_q_chi_eff_sliced_w1'))
axs[0].set(xlabel=r'Effective-spin NW1',ylabel='Joint sliced W1',title='(a) Joint disagreement')
for r in joint:
    _,m,c=stage(r)
    axs[1].plot([0,1],[float(r['chi_p_nw1']),float(r['chi_eff_nw1'])],color=c,marker=m,ms=3,lw=.7)
clean(axs[1]); axs[1].set_xticks([0,1],[r'$\chi_p$',r'$\chi_{\rm eff}$'])
axs[1].set(xlim=(-.2,1.2),ylabel='NW1',title='(b) Spin comparison')
handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
save(fig,'fig_joint_mechanism',joint)

rep_ids={r['event'] for r in joint if r['stage']=='replication'}
rep=[r for r in events if r['event'] in rep_ids]
assert len(rep)==9
fig,ax=plt.subplots(figsize=(6.5,3.3),layout='constrained')
for r in rep:
    ax.plot([0,1],[float(r['luminosity_distance_nw1']),float(r['chi_eff_nw1'])],color='.25',lw=.9,marker='o',ms=4)
clean(ax); ax.set_xticks([0,1],['Luminosity distance',r'Effective spin $\chi_{\rm eff}$'])
ax.set(xlim=(-.2,1.2),ylabel='NW1')
save(fig,'fig_h3_prospective_replication',rep)

research=ROOT/'work/GWTC5_population_extension'
run=research/'results/preflight/reduced_predictive_density_20261014T013218951048Z'
summary=bind(run/'summary.json'); assert sha(summary)=='ccc11389a6f76f3a003f1ce7a515ea35f7ba82d5b59bfe400a3f5dd46aafd24c'
j=json.loads(summary.read_text()); entry=j['levels'][-1]['arrays']; a=bind(research/entry['path']); assert sha(a)==entry['sha256']
with np.load(a,allow_pickle=False) as f:
    x=f['chi_fine'].copy(); ys=f['curves_fine'][:,:2].copy()
fig,ax=plt.subplots(figsize=(6.5,3.25),layout='constrained')
for i,(name,style,color) in enumerate([('ST','-','0.1'),('XPNR','--','0.45')]):
    ax.plot(x,ys[:,i],label=name,ls=style,color=color,lw=1.5)
ax.set(xlim=(-.99,.99),ylim=(0,None),xlabel=r'Effective spin $\chi_{\rm eff}$',ylabel='Population predictive density')
clean(ax); ax.set_xticks([-.99,-.5,0,.5,.99]); ax.legend(frameon=False)
save(fig,'fig_population_predictive',{'chi':x.tolist(),'curves':ys.tolist()})

manifest=bind(research/'results/preflight/reduced_evidence_handoff_20261014T010712696648Z/figures_v2/figure_manifest.json')
j=json.loads(manifest.read_text()); vals=np.array(j['plotted_continuous_values']); names=j['plotted_group_names']
assert vals.shape==(11,2,2)
fig,axs=plt.subplots(1,2,figsize=(6.5,4.05),sharey=True,layout='constrained')
names=['Full rows','First half','First three quarters',*[f'Contiguous quarter {i}' for i in range(4)],*[f'Interleaved quarter {i}' for i in range(4)]]
for f,ax in enumerate(axs):
    for b,(name,m,c,offset) in enumerate([('ST','o','.1',-.12),('XPNR','D','.45',.12)]):
        ax.scatter(vals[:,b,f],np.arange(11)+offset,marker=m,facecolors='none',edgecolors=c,s=20,label=name,zorder=3)
        ax.axvline(vals[0,b,f],ls=':' if b==0 else '--',color=c,lw=.8)
    ax.set_yticks(range(11),names); ax.tick_params(axis='y',length=0)
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(axis='x',color='.88',lw=.5); ax.margins(x=.12,y=.05)
axs[0].invert_yaxis(); axs[1].legend(frameon=False,loc='lower right')
axs[0].set(xlabel='Expected physical\npopulation mean',title='(a) Mean')
axs[1].set(xlabel='Expected intrinsic\npopulation SD',title='(b) Intrinsic width')
save(fig,'fig_population_rows',{'values':vals.tolist(),'groups':names})

# Empirical CDFs use every original finite sample, without density smoothing.
processed=ROOT/'work/gwtc5_h1_rerun/GWTC5_reproducible_archive_v1_0_0/data/processed'
cases=[('GW240630_101703','chi_eff',.1433495241632629),('GW240910_103535','chirp_mass',.15119771828335038),('GW241101_220523','chirp_mass',.007741618479245755)]
index={}
for p in processed.rglob('*_posteriors.hdf5'):
    with h5py.File(p) as f:
        for event in f: index[event]=p
fig,axs=plt.subplots(1,3,figsize=(6.5,2.3),sharey=True,layout='constrained')
cdf_record=[]
for k,(event,param,expected) in enumerate(cases):
    p=bind(index[event]); arr=[]
    with h5py.File(p) as f:
        for model in ['C00:IMRPhenomXPHM-SpinTaylor','C00:IMRPhenomXPNR']:
            g=next(g for g in f[event].values() if str(g.attrs['original_label'])==model)
            a=np.array(g[param][:],dtype=float); arr.append(a[np.isfinite(a)])
    widths=[np.quantile(a,.95)-np.quantile(a,.05) for a in arr]
    nw=wasserstein_distance(*arr)/np.mean(widths)
    assert abs(nw-expected)<5e-12
    for b,a in enumerate(arr):
        xs=np.sort(a); ys=np.arange(1,len(a)+1)/len(a)
        axs[k].plot(xs,ys,color='.1' if b==0 else '.5',ls='-' if b==0 else '--',lw=1.1,label=['ST','XPNR'][b])
    clean(axs[k]); axs[k].set_ylim(0,1.03)
    axs[k].set_title(f'({chr(97+k)}) {event}',fontsize=8)
    axs[k].set_xlabel(r'$\chi_{\rm eff}$' if param=='chi_eff' else 'Detector chirp mass\n'+r'($M_\odot$)')
    axs[k].text(.04,.94,f'NW1 = {nw:.4f}',transform=axs[k].transAxes,fontsize=8,va='top')
    cdf_record.append({'event':event,'parameter':param,'nw1':nw,'rows':[len(a) for a in arr],'sample_sha256':[hashlib.sha256(a.astype('<f8').tobytes()).hexdigest() for a in arr]})
axs[0].set_ylabel('Cumulative probability'); axs[-1].legend(loc='lower right',frameon=False,fontsize=8)
save(fig,'fig_cdf_examples_readable',cdf_record)

for rel,digest in inputs.items(): assert sha(ROOT/rel)==digest
(OUT/'figure_manifest.json').write_text(json.dumps({'inputs':inputs,'outputs':outputs,'new_fit':False,'source_data_changed':False,'all_78_events_retained':True,'cdf_values_checked':True,'vector_pdf_and_svg':True,'png_dpi':450},indent=2)+'\n')
print(json.dumps({'figures':len(outputs)//4,'output':str(OUT),'inputs_unchanged':True}))
