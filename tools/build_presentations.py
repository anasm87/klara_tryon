"""Evidence-led HTML decks. Photos remain unaltered; charts use recorded case means."""
from pathlib import Path
import json, base64, html, shutil, hashlib
import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import find_contours

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT
OUT=ROOT/'docs'
E=SOURCE/'evidence/final-test-1000-01'
A=json.loads((E/'analysis.json').read_text()); RESULT=json.loads((E/'results.json').read_text())
RECORD={(r['case_id'],r['variant'],r['seed']):r for r in RESULT['records']}
for part in ('mvp','final','report','private'): (OUT/part).mkdir(parents=True,exist_ok=True)
INK='#15353b';GREEN='#137f72';RED='#b94e39';MUTED='#566d70';PAPER='#f7f4ed';GOLD='#bf8b43'
def b64(p):
    return 'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()
def ref(sid,role):return E/'references'/sid/(role+'.png')
def generated(sid,variant,seed=9026):return E/RECORD[sid,variant,seed]['file']
def photo(p,label,extra=''):
    return f'<figure {extra}><img src="{b64(p)}" alt="{html.escape(label)}"><figcaption>{label}</figcaption></figure>'
def note(s):return f'<p class="small">{s}</p>'
def line(x1,y1,x2,y2,color=MUTED,width=2):return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'
def text(x,y,s,size=24,color=INK,anchor='start',weight=400):return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(s))}</text>'
def svg(body,w=1100,h=430):return f'<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {w} {h}" style="width:100%;height:100%;font-family:Arial,sans-serif">{body}</svg>'
def network():
    boxes=[(0,130,180,105,'VAE encoder','Frozen',MUTED),(260,100,500,175,'U-Net','',INK),(840,130,215,105,'VAE decoder','Frozen',MUTED)]
    s=''
    for x,y,w,h,label,sub,col in boxes:
        s+=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{col}" stroke-width="2"/>'+text(x+w/2,y+43,label,27,col,'middle',600)
        if sub:s+=text(x+w/2,y+78,sub,21,col,'middle')
    s+=line(180,182,260,182)+line(760,182,840,182)
    s+=text(90,73,'Person + garment',21,INK,'middle')+line(90,88,90,130)
    s+=f'<rect x="298" y="164" width="424" height="78" fill="{GREEN}"/>'
    s+=text(510,194,'16 self-attention modules',25,'white','middle',600)+text(510,226,'Q / K / V / output projections',21,'white','middle')
    s+=text(510,322,'49.6 million trainable parameters',29,GREEN,'middle',600)
    s+=text(510,360,'Other U-Net weights stay frozen',24,MUTED,'middle')
    s+=text(947,300,'Generated image',23,INK,'middle')+line(947,235,947,270)
    s+=text(775,77,'U-Net + scheduler during generation',20,MUTED,'middle')
    return svg(s,1080,390)
def splitbar():
    s='';x=0
    for n,label,c in [(1000,'Train',GREEN),(64,'Validation',GOLD),(32,'Test',RED)]:
        w=1100*n/1096;s+=f'<rect x="{x}" y="0" width="{w}" height="24" fill="{c}"/>';x+=w
    for x,n,label,col in [(0,1000,'training cases',GREEN),(430,64,'validation cases',GOLD),(785,32,'final-test cases',RED)]:
        s+=text(x,82,f'{n:,}',44,col,weight=700)+text(x+130,80,label,24)
    return svg(s,1100,105)
def paired():
    s='';x1,x2=165,505;y0,y1=375,30;maximum=.3
    yy=lambda v:y0-v/maximum*(y0-y1)
    for v in [0,.1,.2,.3]:s+=line(95,yy(v),550,yy(v),'#d9dfd9',1)+text(82,yy(v)+7,f'{v:.1f}',21,MUTED,'end')
    for sid in A['complete_cases']:
        p=A['case_means'][sid]['pretrained']['garment_mae'];q=A['case_means'][sid]['adapted-1000']['garment_mae'];c=GREEN if q<p else RED
        s+=f'<g opacity=".48">'+line(x1,yy(p),x2,yy(q),c,2)+f'<circle cx="{x1}" cy="{yy(p)}" r="4" fill="{c}"/><circle cx="{x2}" cy="{yy(q)}" r="4" fill="{c}"/></g>'
    s+=text(x1,414,'Pretrained',24,INK,'middle')+text(x2,414,'Fine-tuned',24,INK,'middle')
    s+=text(95,460,'Each line is one case. Lower error is better.',22,MUTED)
    return svg(s,650,480)
def scatter():
    # Domain deliberately contains every final-test paired change.
    s='';xl,xr,yt,yb=90,710,38,370
    xx=lambda v:xl+(v+.12)/.16*(xr-xl)
    yy=lambda v:yb-(v+.085)/.11*(yb-yt)
    for v in [-.12,-.08,-.04,0,.04]:s+=line(xx(v),yt,xx(v),yb,'#dce0db',1)+text(xx(v),yb+29,f'{v:.2f}',20,MUTED,'middle')
    for v in [-.08,-.04,0,.02]:s+=line(xl,yy(v),xr,yy(v),'#dce0db',1)+text(xl-12,yy(v)+6,f'{v:.2f}',20,MUTED,'end')
    s+=line(xx(0),yt,xx(0),yb,MUTED,2)+line(xl,yy(0),xr,yy(0),MUTED,2)
    for sid in A['complete_cases']:
        p=A['case_means'][sid]['pretrained'];q=A['case_means'][sid]['adapted-1000'];x=q['garment_mae']-p['garment_mae'];y=q['outside_mae_input']-p['outside_mae_input'];c=GREEN if x<0 and y<0 else RED
        s+=f'<circle cx="{xx(x)}" cy="{yy(y)}" r="6" fill="{c}"><title>{sid}: garment {x:.5f}, outside {y:.5f}</title></circle>'
    s+=text(115,330,'11 improve both',23,GREEN,weight=600)
    s+=text(400,440,'Change in garment error',24,INK,'middle')
    s+=f'<text transform="translate(24 215) rotate(-90)" fill="{INK}" font-size="23" text-anchor="middle">Change in outside-clothing error</text>'
    return svg(s,760,470)
def maskview(sid='12219_00'):
    folder=E/'references'/sid
    masks={r:np.asarray(Image.open(folder/(r+'.png')).convert('L'))>=128 for r in ('edited_mask','target_mask')}
    garment=np.asarray(Image.fromarray((masks['target_mask']*255).astype('uint8')).filter(ImageFilter.MinFilter(7)))>0
    garment[:3,:]=False;garment[-3:,:]=False;garment[:,:3]=False;garment[:,-3:]=False
    outside=np.asarray(Image.fromarray(((masks['edited_mask']|masks['target_mask'])*255).astype('uint8')).filter(ImageFilter.MaxFilter(17)))==0
    s=f'<image href="{b64(folder/"target.png")}" width="384" height="512"/>'
    for mask,col,opacity in [(outside,GOLD,.18),(garment,GREEN,.36)]:
        contours=find_contours(np.pad(mask.astype(float),1),.5)
        paths=[]
        for contour in contours:
            points=' L '.join(f'{x-1:.1f},{y-1:.1f}' for y,x in contour[::3])
            paths.append('M '+points+' Z')
        s+=f'<path d="{" ".join(paths)}" fill-rule="evenodd" fill="{col}" fill-opacity="{opacity}" stroke="{col}" stroke-width="1"/>'
    return svg(s,384,512)

CSS='''*{box-sizing:border-box}html,body{margin:0;background:#152c31;color:#15353b;font-family:Arial,sans-serif}body{overflow:hidden}.slide{width:1280px;height:720px;background:#f7f4ed;position:relative;padding:45px 58px;overflow:hidden;display:none;transform-origin:top left}.slide.active{display:block}.slide.dark{background:#15353b;color:#f7f4ed}.slide h1{font-size:52px;line-height:1.06;letter-spacing:-1.7px;margin:0 0 25px;font-weight:700}.slide h2{font-size:40px;line-height:1.12;letter-spacing:-.8px;margin:0 0 24px}.slide p{font-size:26px;line-height:1.38;margin:12px 0}.slide .small{font-size:18px;line-height:1.4;color:#566d70}.slide.dark .small{color:#b4cac7}.eyebrow{font-size:17px;letter-spacing:2.4px;text-transform:uppercase;margin-bottom:17px;color:#137f72}.dark .eyebrow{color:#95d7c2}.footer{position:absolute;left:58px;right:58px;bottom:24px;display:flex;justify-content:space-between;font-size:14px;color:#566d70}.dark .footer{color:#b4cac7}.row{display:flex;gap:28px}.photos{display:flex;gap:24px;align-items:start}.photos figure{flex:1;min-width:0;margin:0}.photos img{display:block;width:100%;height:330px;object-fit:contain;background:#e8e8e2}.photos figcaption{font-size:22px;margin-top:10px;color:inherit}.photos.compact img{height:270px}.photos.compact figcaption{font-size:20px}.green{color:#137f72}.rust{color:#b94e39}.metric{font-size:78px;line-height:1.05;font-weight:700;letter-spacing:-3px}.rule{border-top:1px solid #b7c2bc;padding-top:18px}.bigquote{font-size:44px!important;line-height:1.15!important;letter-spacing:-1px}.controls{position:fixed;bottom:12px;right:18px;display:flex;gap:8px;z-index:5}.controls button{background:#fff;border:1px solid #aaa;padding:9px 14px;font-size:14px;cursor:pointer}aside.notes{display:none;position:fixed;inset:12% 8%;background:#fff;padding:30px;z-index:8;font-size:22px;line-height:1.5;overflow:auto}aside.notes.open{display:block}.note-title{font-size:16px;text-transform:uppercase;letter-spacing:2px;color:#137f72}table{border-collapse:collapse;width:100%;font-size:23px}td,th{padding:18px 10px;border-bottom:1px solid #cbd3cc;text-align:left}th{font-size:19px;color:#566d70}a{color:#137f72}.dark a{color:#95d7c2}@page{size:13.333333in 7.5in;margin:0}@media print{html,body{background:white;overflow:visible}.slide{display:block!important;transform:none!important;page-break-after:always;break-after:page}.slide:last-of-type{page-break-after:auto}.controls,aside.notes{display:none!important}}'''
slides={};notes={}
def add(deck,title,body,script,seconds,source,dark=False):
    slides.setdefault(deck,[]);notes.setdefault(deck,[]);n=len(slides[deck])+1
    notes[deck].append({'slide':n,'title':title,'seconds':seconds,'script':script,'source':source})
    footer=f'<footer class="footer"><span>Anas Mhana / Klara</span><span>{n:02}</span></footer>'
    slides[deck].append(f'<section class="slide {"dark" if dark else ""}" aria-label="{html.escape(title)}">{body}{footer}</section>')
SRC='VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use.'

# MVP: a method-focused presentation consistent with the diary, without false status claims.
sid='07703_00'
opening=f'''<div class="eyebrow">Klara / MVP / 6 October 2026</div><div class="row"><div style="width:420px;padding-top:35px"><h1>Would you trust<br>this preview?</h1><p>The shirt looks plausible.<br>Its color has changed.</p><p class="small" style="margin-top:52px">A study of garment fidelity in<br>AI virtual try-on</p></div><div class="photos" style="width:700px;margin-top:20px">{photo(ref(sid,'garment'),'Requested garment')}{photo(generated(sid,'pretrained'),'Pretrained preview')}</div></div><p class="small" style="position:absolute;left:505px;top:588px">Selected example: {sid}, seed 9026. No claim about failure frequency.</p>'''
add('mvp','Would you trust this preview?',opening,"Imagine choosing this pink striped shirt online. You ask an AI tool to show it on a person, and the preview looks believable. But the color is different. If the preview changes the item you are thinking of buying, how useful is it? That is the problem I am investigating with Klara.",35,SRC)
sid='12219_00'
body=f'''<div class="eyebrow">The research question</div><h2>Can attention fine-tuning improve the garment<br>while preserving the person and background?</h2><div class="photos compact">{photo(ref(sid,'person'),'Edited person input')}{photo(ref(sid,'garment'),'Requested garment')}{photo(ref(sid,'target'),'Original reference')}</div><p class="small">VITON-HD and VITON-HD-edit provide an aligned reference for evaluation.<br>The intended audience is developers of online clothing-preview tools.</p>'''
add('mvp','The research question',body,"My question is whether fine-tuning the model's attention weights can improve garment reconstruction while preserving the person and background. This is useful to developers building clothing previews. The data gives me an edited person photo, a requested garment and an original reference photo. That lets me compare the generated result with something concrete. It does not tell me whether the garment would physically fit a customer.",40,SRC)
body=f'''<div class="eyebrow">Data and experiment design</div><h2>Separate data for learning and evaluation</h2><div style="height:140px;margin-top:48px">{splitbar()}</div><div class="row" style="margin-top:45px;gap:75px"><div style="width:530px"><h2 style="font-size:29px">Checks before modeling</h2><p>Pairings and image dimensions<br>Missing files and exact duplicates<br>Source IDs across the splits</p></div><div style="width:520px"><h2 style="font-size:29px">A limit I need to keep visible</h2><p>Different source IDs do not prove<br>that every person is different.</p><p class="small">Custom split of the original VITON-HD test partition.<br>Upstream training exposure is unverified.</p></div></div>'''
add('mvp','Separate data for learning and evaluation',body,"The experiment uses one thousand training cases, sixty-four validation cases and thirty-two final-test cases. The roles are separate: training changes the weights, validation helps development, and the final test checks the fixed model. I check pairings, file quality and exact overlap. A limitation is that different source IDs do not guarantee different people. These are custom splits, so I will not present them as official benchmark results.",40,'Frozen experiment-plan.json and final-evaluation-plan.json. Split checks and provenance receipts.')
body=f'''<div class="eyebrow">The intervention</div><h2>Fine-tuning the attention already in the model</h2><div style="height:380px;margin-top:25px">{network()}</div><p class="rule">The experiment changes selected attention weights.<br>The VAE and other U-Net weights stay frozen.</p>'''
add('mvp','Fine-tuning existing attention',body,"I use the pretrained CatVTON-MaskFree model. I am focusing the experiment on existing self-attention projections inside its U-Net. Attention lets spatial features interact, including information from the person and garment. The VAE and the remaining U-Net weights stay frozen. The comparison is therefore between the original checkpoint and the attention-adapted version, rather than between unrelated applications.",40,'CatVTON-MaskFree pinned vendor source. configure_attention in training_core.py.')
body=f'''<div class="eyebrow">Evaluation plan</div><h2>A convincing image can still be wrong</h2><div class="row"><div style="width:330px;height:440px">{maskview()}</div><div style="width:740px;padding:18px 0 0 40px"><h2 style="font-size:30px;color:{GREEN}">Inside the clothing region</h2><p>MAE measures pixel differences.<br>SSIM measures structural agreement.</p><h2 style="font-size:30px;color:{GOLD};margin-top:40px">Outside the clothing region</h2><p>Measure unwanted changes<br>relative to the input person.</p><p class="small" style="margin-top:25px">Same cases, seeds and settings for both models.<br>Masks define measurement regions, not generator inputs.</p></div></div>'''
add('mvp','Evaluation plan',body,"A realistic-looking image is not enough. I measure garment pixel error with MAE and structural agreement with SSIM. Outside the clothing region, I measure changes relative to the input. The colored areas show actual evaluation regions. Both models use the same cases, seeds and generation settings. I compare results case by case and inspect failures, because one average can hide very different outcomes.",45,'Exact regional masks and metrics in analyze_edit_evaluation.py. Green: eroded target clothing. Gold: outside dilated source/target union.')
body='''<div class="eyebrow">MVP discussion</div><h1>The feedback I need</h1><p class="bigquote" style="max-width:1050px;margin-top:42px">Do these measurements answer<br>the research question clearly?</p><p style="margin-top:40px">What should I explain more simply<br>for a mixed technical and nontechnical jury?</p><div class="rule" style="margin-top:45px"><p>After the MVP: review the method and findings,<br>then prepare the final story and demonstration.</p><p class="small">Rehearsal: 14 October. Jury: 15 October.</p></div>'''
add('mvp','The feedback I need',body,"For this MVP, I want feedback on the research question and how I explain the evaluation. Do these measurements cover the important part of the problem? And what would you simplify for the final jury? I will use that feedback to refine the scientific story and choose clear examples for the final presentation.",30,'DS#055 schedule and KLARA_DAILY_PROGRESS.md. Method-focused presentation scope, not a claim of unfinished execution.',True)

# Jury: eight main slides, 300 seconds. The final-test numbers appear only here.
add('final','Would you trust this preview?',opening.replace('MVP / 6 October 2026','Final project / 15 October 2026'),"Imagine choosing this pink striped shirt online. You ask an AI tool for a preview, and the result looks believable. But the color has changed. Would that help you decide what to buy? Klara started with this problem: a convincing image can still show the wrong clothing.",25,SRC)
sid='12219_00'
body=f'''<div class="eyebrow">Question and data</div><h2>Better garment reconstruction,<br>without unwanted changes elsewhere?</h2><div class="photos compact">{photo(ref(sid,'person'),'Edited person input')}{photo(ref(sid,'garment'),'Requested garment')}{photo(ref(sid,'target'),'Original reference')}</div><div style="height:102px;margin-top:19px">{splitbar()}</div><p class="small">VITON-HD + VITON-HD-edit. Custom reconstruction study for clothing-preview developers.</p>'''
add('final','Question and data',body,"My question was whether attention fine-tuning could improve garment reconstruction while preserving the person and background. I used VITON-HD and VITON-HD-edit. The edited person and catalog garment are inputs. The original photo is the reference. I separated one thousand training cases, sixty-four validation cases and thirty-two final-test cases. This is a reconstruction study for clothing-preview developers. It does not test sizing or physical fit, and identity separation remains unverified.",40,'Data: NXN-Labs/VITON-HD-edit revision 9de94ee10e15f5069fd650ce4c914215f124eb6f. Frozen custom splits. No official benchmark claim.')
body=f'''<div class="eyebrow">What changed</div><h2>I adapted the existing attention weights</h2><div style="height:375px;margin-top:15px">{network()}</div><div class="rule"><p>1,000 cases, three epochs, 3,000 optimizer updates</p><p class="small">AdamW, learning rate 0.00001. Training objective: predicted-noise MSE.</p></div>'''
add('final','What changed in the model',body,"The starting point was CatVTON-MaskFree. I adapted the query, key, value and output projections in sixteen existing self-attention modules. The VAE and other U-Net weights stayed frozen. Training covered all one thousand cases in three epochs, with three thousand optimizer updates. The training loss compares predicted noise with the noise added to the target. The image metrics come later, during evaluation. Checks confirmed that frozen weights stayed unchanged and the checkpoint reloaded correctly.",35,'run.json and steps.jsonl. 49,574,080 selected parameters. Standard epsilon-MSE adaptation; not a reproduction of DREAM training.')
body=f'''<div class="eyebrow">A fair comparison</div><h2>Same inputs. Same settings. Two models.</h2><div class="row"><div style="width:315px;height:420px">{maskview('13008_00')}</div><div style="width:780px;padding-left:34px"><p class="green" style="font-size:32px;font-weight:bold">Garment MAE and SSIM</p><p>How close is the garment to the reference?</p><p style="color:{GOLD};font-size:32px;font-weight:bold;margin-top:33px">Outside-clothing MAE</p><p>How much of the input person changed?</p><p class="rule" style="margin-top:32px">32 cases × 2 models × 2 seeds</p><p class="small">128 attempts, 122 images, 27 complete paired cases.<br>Six safety exclusions across five cases. No rerolls.</p></div></div><p class="small">384 × 512 pixels, 20 DDIM steps, guidance 2.5. Average seeds within each case.</p>'''
add('final','A fair comparison',body,"I fixed the checkpoint before the final test. Both models saw the same inputs and two fixed seeds, with identical generation settings. Garment MAE is the primary metric. SSIM checks structure, and outside-clothing error checks preservation. The masks define measurement regions; the generator receives no mask. There were one hundred twenty-eight attempts. Six safety exclusions left twenty-seven complete paired cases. I kept those exclusions and averaged the two seeds within each case.",40,'Frozen final-evaluation-plan.json. results.json: 128 attempts, 122 saved outputs. Six exclusions affect five cases.')
body=f'''<div class="eyebrow">Final-test finding</div><h2>Garment error fell 15.6% on average</h2><div class="row"><div style="width:690px;height:475px">{paired()}</div><div style="width:410px;padding-top:30px"><div class="metric green">21 / 27</div><p>cases improved</p><p class="rule" style="margin-top:30px">Mean MAE<br><b>0.1335</b> pretrained<br><b class="green">0.1127</b> fine-tuned</p><p class="small">Green lines improve.<br>Six red lines worsen.</p></div></div><p class="small">Paired mean difference: −0.02085. Descriptive 95% case-bootstrap interval: [−0.03222, −0.01032].</p>'''
add('final','Garment reconstruction improved on average',body,"The fine-tuned model reduced average garment error by fifteen point six percent. Each line here is a test case. Green lines improve, and red lines worsen. Twenty-one of the twenty-seven complete cases improved. The mean fell from about zero point one three four to zero point one one three. The paired bootstrap interval was below zero, but it describes uncertainty across these cases. It does not establish performance across new populations or different training seeds.",45,'analysis.json, garment_mae. Equal case weights after averaging two seeds. 10,000 paired case-bootstrap resamples, seed 41026.')
body=f'''<div class="eyebrow">The trade-off</div><h2>Preservation did not improve for every person</h2><div class="row"><div style="width:800px;height:475px">{scatter()}</div><div style="width:335px;padding-top:20px"><div class="metric rust">14 / 27</div><p>cases had more<br>outside-clothing error</p><p class="rule" style="margin-top:35px">Yet average outside error<br><b>fell 25.9%</b>.</p><p class="small">Larger gains in some cases<br>outweighed smaller regressions.</p></div></div><p class="small">Each dot is one paired case. Differences are fine-tuned minus pretrained. Lower left improves both errors.</p>'''
add('final','Preservation remained inconsistent',body,"Preservation tells a more complicated story. Each dot is one case. Moving left means better garment reconstruction, and moving down means better preservation. Eleven cases improved on both. But fourteen had more outside-clothing error after fine-tuning, even though average outside error fell. Some larger improvements outweighed smaller regressions. This is why I would not describe the model as reliably preserving every person based on the average alone.",40,'analysis.json, case_means and joint_outcomes. Outside-input MAE: 13 improvements, 14 regressions; relative mean change −25.899%.')
sid='08931_00'
body=f'''<div class="eyebrow">Looking beyond the average</div><h2>Even the largest gain still changes the person</h2><div class="photos" style="margin-top:28px">{photo(ref(sid,'person'),'Input person')}{photo(generated(sid,'pretrained'),'Pretrained')}{photo(generated(sid,'adapted-1000'),'Fine-tuned')}{photo(ref(sid,'target'),'Reference')}</div><div class="row rule" style="margin-top:27px"><p style="width:550px" class="green">Large clothing artifacts are reduced.</p><p style="width:560px" class="rust">The arm pose and trousers still change.</p></div><p class="small">Case 08931_00, seed 9026. Selected by the largest case-average garment-MAE reduction. Full gallery includes all cases.</p>'''
add('final','A metric gain can still contain visible errors',body,"This case had the largest reduction in garment error, so it is a useful test of what the number means. The fine-tuned image removes much of the severe clothing distortion. But compare it with the input and reference: the arm is raised and the trousers still change. The improvement is real under the metric, while the image remains imperfect. I kept all cases and both seeds in the gallery so the audience can inspect more than a few selected successes.",45,SRC+' Selection rule: smallest case-average fine-tuned minus pretrained garment MAE. Assistant observations are qualitative, not human-rating scores.')
body='''<div class="eyebrow">Answer and recommendation</div><h1>Better reconstruction.<br>Preservation remains a challenge.</h1><p style="max-width:1080px;margin-top:28px">Attention fine-tuning helped on this dataset.<br>It did not make every clothing preview dependable.</p><div class="row rule" style="margin-top:38px"><div style="width:660px"><p>Use Klara as a research demonstration.<br>Show the failures alongside the successes.</p><p class="small">Limits: edited inputs, 27 complete pairs, one training seed.<br>Identity separation and upstream exposure are unverified.</p></div><div style="width:400px"><p><a href="https://klara-app.de">klara-app.de</a></p><p class="small">Live demo uses the trained checkpoint.<br>Recorded gallery works offline.</p></div></div>'''
add('final','Answer to the research question',body,"My conclusion is that attention fine-tuning improved average reconstruction on this dataset, while preservation remained inconsistent. I would use Klara as a research demonstration with visible failures. The small complete-case test, edited inputs and one training seed limit broader claims. The website uses the trained checkpoint, and the offline gallery preserves the experiment. The next research step would be broader independent data and repeated training seeds.",30,'Final analysis and limitations. Website configuration differs: 768×1024, 50 steps versus research 384×512, 20 steps.',True)
assert sum(n['seconds'] for n in notes['final'])==300

SCRIPT='''const slides=[...document.querySelectorAll('.slide')];let current=0;const notes=NOTES;function resize(){const s=slides[current];s.style.transform=`scale(${Math.min(innerWidth/1280,innerHeight/720)})`;}function show(n){slides[current].classList.remove('active');slides[current].style.transform='';current=(n+slides.length)%slides.length;slides[current].classList.add('active');document.querySelector('#note-body').textContent=notes[current].script;document.querySelector('#count').textContent=`${current+1}/${slides.length}`;resize();}document.querySelector('#next').onclick=()=>show(current+1);document.querySelector('#prev').onclick=()=>show(current-1);document.querySelector('#notes').onclick=()=>document.querySelector('aside').classList.toggle('open');document.querySelector('#full').onclick=()=>document.documentElement.requestFullscreen();addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(e.key))show(current-1);if(e.key==='n')document.querySelector('aside').classList.toggle('open');if(e.key==='Escape')document.querySelector('aside').classList.remove('open')});addEventListener('resize',resize);show(0);'''
for deck in slides:
    payload='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Klara '+deck+'</title><style>'+CSS+'</style></head><body>'+''.join(slides[deck])+'<div class="controls"><button id="prev">Previous</button><button id="count"></button><button id="next">Next</button><button id="notes">Notes (N)</button><button id="full">Fullscreen</button></div><aside class="notes"><p class="note-title">Speaker notes (N to close)</p><p id="note-body"></p></aside><script>const NOTES='+json.dumps(notes[deck])+';'+SCRIPT+'</script></body></html>'
    (OUT/deck/('Klara_'+deck.title()+'.html')).write_text(payload,encoding='utf-8')
    (OUT/deck/'slides.json').write_text(json.dumps(notes[deck],indent=2),encoding='utf-8')
    (OUT/deck/'SPEAKER_NOTES.md').write_text('# Klara '+deck.upper()+' speaker notes\n\nUse these as a rehearsal script. Speak naturally and adjust wording to your voice. '+('Planned duration: five minutes.' if deck=='final' else 'Method-focused MVP. Confirm the allotted time with the instructor.')+'\n\n'+'\n\n'.join(f"## {n['slide']}. {n['title']} ({n['seconds']} seconds)\n\n{n['script']}\n\nSource: {n['source']}" for n in notes[deck]),encoding='utf-8')
print('Built six-slide MVP and eight-slide final presentation with editable SVG evidence and notes.')
(OUT/'final/paired-garment-error.svg').write_text(paired(),encoding='utf-8')
(OUT/'final/preservation-tradeoff.svg').write_text(scatter(),encoding='utf-8')
