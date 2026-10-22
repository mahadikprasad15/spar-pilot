"""Offline review pages; blind exports receive no private source mapping."""
import json

from pilot_eval.scoring import _NUMBER


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'">
<title>Answer review</title><style>
:root{color-scheme:light;--ink:#253e44;--muted:#607476;--line:#d5e0d9;--green:#2e6c60}
*{box-sizing:border-box}body{margin:0;background:#f5f6f0;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}
header,main{max-width:1220px;margin:auto;padding:24px 32px}header{padding-bottom:12px}h1{font:36px/1.2 Georgia,serif;margin:10px 0}h2{font-size:15px;margin:0 0 12px}p{max-width:900px;margin:8px 0}
.small{font-size:13px;color:var(--muted)}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0}.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(280px,1fr);gap:20px}
article,aside{background:#fffefa;border:1px solid var(--line);border-radius:10px;padding:24px;min-width:0}aside{position:sticky;top:16px;align-self:start}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.65 system-ui;margin:0 0 22px}
label{display:block;font-size:13px;font-weight:650;margin:14px 0 5px}input,select,textarea,button{font:inherit;border:1px solid #b8cbc1;border-radius:6px;padding:9px 11px;background:white;color:var(--ink)}input,textarea,aside select{width:100%}textarea{min-height:85px;resize:vertical}button{cursor:pointer}button:hover{background:#eaf2eb}button:disabled{opacity:.45;cursor:default}.primary{background:var(--green);color:white}.primary:hover{background:#23594e}
:focus-visible{outline:3px solid #78a9b5;outline-offset:2px}#message{min-height:26px;color:#8a4639;font-size:13px}#identity{font:12px monospace;color:var(--muted)}#details{background:#eef2e9;border-radius:6px;padding:14px;display:none}.file{font-weight:400;margin:0}#import{max-width:240px;width:auto;font-size:12px}#progress{font-weight:650}#jump{max-width:175px}
@media(max-width:800px){header,main{padding:20px 16px}.grid{grid-template-columns:1fr}aside{position:static}h1{font-size:30px}}
</style></head><body><header><div class="small">Saved-response review · offline</div><h1 id="title">Intended-answer review</h1>
<p id="instructions">Identify the response’s intended final answer. Use absent or ambiguous when appropriate. This does not ask you to validate its reasoning.</p>
<p class="small">Download your draft regularly. Browser storage is a convenience cache, not the durable record. After saving through the notebook, import its updated draft before continuing. Source files are held separately; this page is not an access-control boundary.</p>
<div class="toolbar"><button id="previous">← Previous</button><button id="next">Next →</button><button id="pending">Next pending</button><select id="jump" aria-label="Go to response"></select><span id="progress" aria-live="polite"></span></div>
<div class="toolbar"><button id="download" class="primary">Download draft</button><label class="file" for="import">Import saved draft</label><input id="import" type="file" accept="application/json,.json"></div><div id="message" role="status"></div></header>
<main><div class="grid"><article><div id="identity"></div><h2>Question</h2><pre id="question"></pre><h2>Response</h2><pre id="response"></pre><pre id="details"></pre></article>
<aside><label for="reviewer">Reviewer name or ID</label><input id="reviewer" autocomplete="off"><label for="status" id="status-label">Intended answer</label><select id="status"></select>
<div id="answer-section"><label for="answer">Numeric transcription</label><input id="answer" autocomplete="off" placeholder="e.g. 12, -0.5, or 3/2"><p class="small">Enter one number without currency or units. You may transcribe a written-out number; record the supporting text below.</p></div>
<label for="evidence">Supporting response text</label><textarea id="evidence"></textarea><label for="note">Reviewer note</label><textarea id="note"></textarea><p class="small" id="saved">Edits are cached locally. Download to preserve them.</p></aside></div></main>
<script id="review-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent);
const el=id=>document.getElementById(id),isAnswers=data.mode==='answers';
const key=row=>row.response_id+(isAnswers?'':'|'+row.scorer);
const known=new Map(data.rows.map(row=>[key(row),row]));
const cacheKey='answer-review:'+data.packet_sha256+':'+(data.reveal_sha256||'initial');
const validNumber=value=>new RegExp('^(?:'+data.number_pattern+')$').test(value)&&!/\/0+$/.test(value);
let index=0,state={reviewer:data.reviewer||'',base_revision:0,entries:{}};
const blank=row=>isAnswers?{response_id:row.response_id,status:'pending',answer:null,evidence:'',note:''}:
 {response_id:row.response_id,scorer:row.scorer,category:'pending',evidence:'',note:''};
function notify(text){el('message').textContent=text;}
function restore(payload){
 if(!payload||payload.packet_sha256!==data.packet_sha256)throw Error('This draft belongs to another response packet.');
 if(!isAnswers&&payload.reveal_sha256!==data.reveal_sha256)throw Error('This draft belongs to another revealed result.');
 const rows=isAnswers?payload.labels:payload.annotations;
 if(!Array.isArray(rows))throw Error('The draft does not contain a valid entry list.');
 const revision=payload.base_revision??0;
 if(isAnswers&&(!Number.isInteger(revision)||revision<0||(payload.revision!==undefined&&payload.revision!==revision)))throw Error('Invalid draft revision.');
 const entries={},seen=new Set();
 for(const row of rows){
  const id=key(row),status=isAnswers?row.status:row.category;
  if(!known.has(id)||seen.has(id)||!data.statuses.includes(status))throw Error('Unknown, duplicate or malformed review entry.');
  if(isAnswers&&status==='numeric'&&(typeof row.answer!=='string'||!validNumber(row.answer.trim())))throw Error('Numeric transcription is unsupported or undefined.');
  if(isAnswers&&status!=='numeric'&&row.answer!=null&&row.answer!=='')throw Error('Nonnumeric status carries an answer.');
  if(typeof(row.evidence??'')!=='string'||typeof(row.note??'')!=='string')throw Error('Evidence and notes must be text.');
  seen.add(id);entries[id]=isAnswers?{response_id:row.response_id,status,answer:status==='numeric'?row.answer.trim():null,evidence:row.evidence||'',note:row.note||''}:
   {response_id:row.response_id,scorer:row.scorer,category:status,evidence:row.evidence||'',note:row.note||''};
 }
 if(payload.reviewer!=null&&typeof payload.reviewer!=='string')throw Error('Reviewer ID must be text.');
 state={reviewer:payload.reviewer||'',base_revision:revision,entries};
}
function current(){const row=data.rows[index],id=key(row);return state.entries[id]||(state.entries[id]=blank(row));}
function exported(validate=true){
 const result={packet_sha256:data.packet_sha256,reviewer:state.reviewer};
 const rows=data.rows.map(row=>state.entries[key(row)]||blank(row));
 if(validate&&!state.reviewer.trim())throw Error('Enter your reviewer name or ID before downloading.');
 if(isAnswers){
  if(validate&&rows.some(row=>row.status==='numeric'&&!validNumber((row.answer||'').trim())))throw Error('Finish the numeric transcription or choose pending before downloading.');
  result.base_revision=state.base_revision;result.labels=rows;
 }else{result.reveal_sha256=data.reveal_sha256;result.annotations=rows.filter(row=>row.category!=='pending');}
 return result;
}
function persist(){try{localStorage.setItem(cacheKey,JSON.stringify(exported(false)));}catch(e){notify('Browser storage is unavailable. Download the draft to preserve edits.');}}
function progress(){const count=data.rows.filter(row=>{const entry=state.entries[key(row)];return entry&&(isAnswers?(entry.status!=='pending'&&(entry.status!=='numeric'||validNumber((entry.answer||'').trim()))):entry.category!=='pending');}).length;
 el('progress').textContent=count+'/'+data.rows.length+' reviewed'+(isAnswers?' · saved revision '+state.base_revision:'');}
function render(){
 const row=data.rows[index],entry=current();
 el('identity').textContent=(index+1)+' / '+data.rows.length+' · '+row.response_id+(row.scorer?' · '+row.scorer:'');
 el('question').textContent=row.question;el('response').textContent=row.response;
 el('details').textContent=row.details||'';el('details').style.display=row.details?'block':'none';
 el('reviewer').value=state.reviewer;el('status').value=isAnswers?entry.status:entry.category;
 el('answer').value=entry.answer||'';el('answer').disabled=!isAnswers||entry.status!=='numeric';
 el('evidence').value=entry.evidence;el('note').value=entry.note;el('jump').value=String(index);
 el('previous').disabled=index===0;el('next').disabled=index===data.rows.length-1;progress();
}
for(const status of data.statuses){const option=document.createElement('option');option.value=status;option.textContent=status.replaceAll('_',' ');el('status').appendChild(option);}
data.rows.forEach((row,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent='Response '+(i+1);el('jump').appendChild(option);});
if(!isAnswers){el('title').textContent='Response explanation review';el('instructions').textContent='Annotate extraction discrepancies using the revealed comparisons. These explanations remain separate from the saved answer labels.';el('status-label').textContent='Explanation category';el('answer-section').style.display='none';}
el('reviewer').addEventListener('input',()=>{state.reviewer=el('reviewer').value;persist();});
el('status').addEventListener('change',()=>{const entry=current();if(isAnswers){entry.status=el('status').value;if(entry.status!=='numeric')entry.answer=null;}else entry.category=el('status').value;persist();render();});
for(const field of ['answer','evidence','note'])el(field).addEventListener('input',()=>{current()[field]=el(field).value;persist();progress();});
el('previous').addEventListener('click',()=>{if(index>0){index--;render();}});
el('next').addEventListener('click',()=>{if(index<data.rows.length-1){index++;render();}});
el('pending').addEventListener('click',()=>{for(let n=1;n<=data.rows.length;n++){const candidate=(index+n)%data.rows.length,entry=state.entries[key(data.rows[candidate])];if(!entry||(isAnswers?entry.status:entry.category)==='pending'){index=candidate;render();return;}}notify('No pending entries remain. Download your draft.');});
el('jump').addEventListener('change',()=>{const choice=Number(el('jump').value);if(Number.isInteger(choice)&&choice>=0&&choice<data.rows.length){index=choice;render();}});
el('download').addEventListener('click',()=>{try{const blob=new Blob([JSON.stringify(exported(),null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=isAnswers?'answers.json':'explanations.json';a.click();URL.revokeObjectURL(url);notify('Draft downloaded. Save it through the notebook to create a durable revision.');}catch(e){notify(e.message);}});
el('import').addEventListener('change',async()=>{try{const file=el('import').files[0];if(!file)return;const payload=JSON.parse(await file.text());if(!confirm('Replace the page draft with this file? Download current edits first if needed.'))return;restore(payload);persist();render();notify('Saved draft imported.');}catch(e){notify(e.message);}});
try{const cached=localStorage.getItem(cacheKey);if(cached)restore(JSON.parse(cached));}catch(e){notify('Cached draft could not be restored: '+e.message);}
render();
</script></body></html>'''


def _page(data):
    embedded=json.dumps(data,ensure_ascii=True).replace('<','\u003c').replace('>','\u003e').replace('&','\u0026')
    return _PAGE.replace('__DATA__',embedded).encode()


def review_page(packet,packet_sha256):
    return _page(dict(mode='answers',packet_sha256=packet_sha256,rows=packet,
        statuses=['pending','numeric','absent','ambiguous'],number_pattern=_NUMBER))


def explanation_page(result,reveal_sha256,categories):
    rows=[]
    for item in result['responses']:
        for scorer in ['strict','flexible_v3','last_number']:
            details=dict(intended_answer=item['label'],gold=item['references'][0]['gold'],
                source_comparisons=[dict(method=r['method'],step=r['step'],capped=r['capped'],
                    comparison=r['comparisons'][scorer]) for r in item['references']])
            rows.append(dict(response_id=item['response_id'],scorer=scorer,question=item['question'],
                response=item['response'],details=json.dumps(details,indent=2,ensure_ascii=False)))
    return _page(dict(mode='explanations',packet_sha256=result['packet_sha256'],reviewer=result['reviewer'],
        reveal_sha256=reveal_sha256,rows=rows,statuses=['pending',*categories],number_pattern=_NUMBER))
