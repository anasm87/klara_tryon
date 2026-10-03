const $ = id => document.getElementById(id);
let inputs={person:null,garment:null},gpu=false,token='',running=false,selecting=0;
const selectionVersion={person:0,garment:0};
function savedJob(value){try{if(value===undefined)return JSON.parse(sessionStorage.getItem('klara-job')||'null');if(value===null)sessionStorage.removeItem('klara-job');else sessionStorage.setItem('klara-job',JSON.stringify(value));}catch{return null;}}
function message(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function enabled(){$('generate').disabled=running||selecting>0||!gpu||!inputs.person||!inputs.garment||!$('permission').checked;}
function read(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});}
function clearResult(){savedJob(null);$('result').hidden=true;$('result').removeAttribute('src');$('download').hidden=true;$('download').removeAttribute('href');$('empty').hidden=false;$('result-tag').textContent='YOUR PREVIEW';$('result-info').textContent='Your result appears after you generate.';}
function preview(kind,data,sampleId=null){clearCase();inputs[kind]=data;$(kind+'-preview').src=data;$(kind+'-preview').hidden=false;$(kind+'-zone').classList.add('has-image');for(const b of $(kind+'-samples').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.id===sampleId));clearResult();enabled();message('Choose your photos, confirm permission, then generate your look.');}
async function api(path,options){const response=await fetch(path,options);if(response.status===401){savedJob(null);location.assign('/signin');throw Error('Please sign in again.');}return response;}
for(const kind of ['person','garment']){$(kind+'-file').addEventListener('change',async event=>{if(running)return;const file=event.target.files[0];if(!file)return;if(!['image/jpeg','image/png','image/webp'].includes(file.type)||file.size>8*1024*1024){message('Choose a JPG, PNG or WebP smaller than 8 MB.',true);event.target.value='';return;}const version=++selectionVersion[kind];selecting++;enabled();try{const data=await read(file);if(version===selectionVersion[kind])preview(kind,data);}catch{message('The image could not be opened.',true);}finally{selecting--;enabled();}});}
$('permission').addEventListener('change',enabled);
async function connection(){try{const response=await api('/api/status');if(!response.ok)throw Error();const state=await response.json();gpu=state.gpu;token=state.token;$('connection').textContent=gpu?'Studio ready':'Preview mode';$('connection').classList.toggle('ready',gpu);if(!gpu)message('Image selection is available. Generation is currently offline.');enabled();}catch{message('Could not connect to the studio. Refresh the page.',true);}}
async function loadSamples(){try{const response=await api('/api/examples');if(!response.ok)throw Error();const catalog=await response.json();for(const kind of ['person','garment']){for(const sample of catalog[kind]){const button=document.createElement('button');button.type='button';button.className='sample-choice';button.dataset.id=sample.id;button.setAttribute('aria-pressed','false');button.title=sample.name;button.setAttribute('aria-label','Choose '+sample.name);const image=document.createElement('img');image.src=sample.url;image.alt=sample.name;image.loading='lazy';button.append(image);const label=document.createElement('span');label.textContent=sample.name;button.append(label);button.addEventListener('click',async()=>{if(running)return;button.disabled=true;const version=++selectionVersion[kind];selecting++;enabled();try{const r=await api(sample.url);if(!r.ok)throw Error();const data=await read(await r.blob());if(version===selectionVersion[kind])preview(kind,data,sample.id);}catch{message('This sample could not be loaded. Please try again.',true);}finally{selecting--;button.disabled=running;enabled();}});$(kind+'-samples').append(button);}}}catch{message('Sample photos are unavailable. You can still upload your own.',true);}}
function lockInputs(locked){for(const control of document.querySelectorAll('.sample-choice,.pair-button,.file-input,#permission,#reset'))control.disabled=locked;}
$('reset').addEventListener('click',()=>{if(running||selecting)return;clearCase();inputs={person:null,garment:null};for(const kind of ['person','garment']){$(kind+'-preview').hidden=true;$(kind+'-preview').removeAttribute('src');$(kind+'-zone').classList.remove('has-image');$(kind+'-file').value='';for(const b of $(kind+'-samples').querySelectorAll('button'))b.setAttribute('aria-pressed','false');}$('permission').checked=false;clearResult();message('Add your photos or choose from the collection.');enabled();});
$('logout').addEventListener('click',async()=>{if(running&&!confirm('Your image will keep generating. Sign out anyway?'))return;try{await api('/api/logout',{method:'POST',headers:{'X-Klara-Token':token}});}finally{savedJob(null);location.assign('/signin');}});
async function generateOrResume(previous=null){
 if(!previous&&$('generate').disabled)return;running=true;enabled();lockInputs(true);$('loading').hidden=false;$('download').hidden=true;$('result').hidden=true;$('empty').hidden=true;$('result-tag').textContent='GENERATING';$('result-info').textContent='';message('Creating your look. You can keep this page open.');
 const start=previous?.started||Date.now();const timer=setInterval(()=>{$('elapsed').textContent=Math.floor((Date.now()-start)/1000)+' seconds elapsed';},1000);
 try{let data=previous;if(!data){const response=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json','X-Klara-Token':token},body:JSON.stringify({...inputs,permission:$('permission').checked})});data=await response.json();if(!response.ok)throw Error(data.error||'Could not start generation.');savedJob({job_id:data.job_id,started:start});}
 let failures=0;for(;;){await new Promise(resolve=>setTimeout(resolve,2000));let state;try{const poll=await api('/api/jobs/'+data.job_id);state=await poll.json();if(!poll.ok)throw Error(state.error||'Could not check job.');failures=0;}catch(error){if(++failures<4)continue;throw Error('Connection lost. The job may still be running; refresh this page to check it.');}
 if(state.status==='error'){savedJob(null);throw Error(state.message);}if(state.status==='done'){savedJob(null);$('result').src=state.image;$('result').hidden=false;$('result-tag').textContent='AI-GENERATED';$('download').href=state.image;$('download').hidden=false;$('result-info').textContent='Created in '+state.seconds+' seconds · 768 × 1024';message('Your look is ready. Take a closer look, then save it.');break;}if(Date.now()-start>20*60*1000)throw Error('This is taking longer than expected. Refresh to check your existing generation.');}
 }catch(error){message(error.message,true);$('result-tag').textContent='NOT COMPLETED';$('empty').hidden=false;}
 finally{clearInterval(timer);$('loading').hidden=true;running=false;lockInputs(false);enabled();}
}
// Only this button starts inference. Reloading an unfinished job only polls it.
$('generate').addEventListener('click',()=>generateOrResume());
connection().then(async()=>{await loadSamples();await loadPresentationExamples();const previous=savedJob();if(previous?.job_id)generateOrResume(previous);});

function clearCase(){const box=$('selected-case');if(box){box.hidden=true;box.textContent='';}for(const b of document.querySelectorAll('.pair-button'))b.setAttribute('aria-pressed','false');}
async function selectPair(example,button){
 if(running)return;
 const versions={person:++selectionVersion.person,garment:++selectionVersion.garment};selecting++;enabled();
 try{
  const values=await Promise.all(['person','garment'].map(async kind=>{const r=await api(example[kind].url);if(!r.ok)throw Error();return read(await r.blob());}));
  if(versions.person!==selectionVersion.person||versions.garment!==selectionVersion.garment)return;
  preview('person',values[0],example.person.id);preview('garment',values[1],example.garment.id);
  button.setAttribute('aria-pressed','true');const box=$('selected-case');box.hidden=false;
  const title=document.createElement('strong');title.textContent=example.person.name+' + '+example.garment.name;box.append(title);
  const note=document.createElement('p');note.textContent='Previous audit: '+example.observation;box.append(note);
  const hint=document.createElement('p');hint.textContent='Inputs selected. Confirm permission, then press Generate to run the trained model.';box.append(hint);
  message('Test pair selected. Press Generate when you are ready.');
  document.querySelector('.workspace').scrollIntoView({behavior:'smooth',block:'start'});
 }catch{message('This test pair could not be loaded. Please try again.',true);}finally{selecting--;enabled();}
}
async function loadPresentationExamples(){
 try{const response=await api('/api/presentation-examples');if(!response.ok)throw Error();const groups=await response.json();
 for(const group of ['better','failure'])for(const example of groups[group]){
  const card=document.createElement('article');card.className='pair-card';
  const pictures=document.createElement('div');pictures.className='pair-pictures';
  for(const kind of ['person','garment']){const im=document.createElement('img');im.src=example[kind].url;im.alt=example[kind].name;im.loading='lazy';pictures.append(im);}card.append(pictures);
  const copy=document.createElement('div');copy.className='pair-copy';const title=document.createElement('h4');title.textContent=example.person.name+' + '+example.garment.name;copy.append(title);
  const note=document.createElement('p');note.textContent=example.observation;copy.append(note);
  const button=document.createElement('button');button.type='button';button.className='pair-button';button.dataset.pair=example.id;button.setAttribute('aria-pressed','false');button.textContent='Use this pair';button.onclick=()=>selectPair(example,button);copy.append(button);card.append(copy);$(group+'-pairs').append(card);
 }}catch{document.querySelector('.examples-intro').textContent='Test pairs could not be loaded. You can still choose individual photos below.';}
}
