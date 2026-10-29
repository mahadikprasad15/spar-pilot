"""Self-contained review client. Downloaded drafts are not confirmed labels."""
import json


def page(packet,packet_sha256):
    # Neither the gold answer nor scorer verdicts are embedded in the client.
    data=json.dumps(dict(packet_sha256=packet_sha256,rows=packet),ensure_ascii=True)
    data=data.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return _PAGE.replace('__DATA__',data)


_PAGE='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'">
<title>Sampling SFT · intended-answer review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f0;color:#253d43;font:16px/1.6 system-ui}
main{max-width:1200px;margin:auto;padding:32px}h1{font:32px Georgia}button,input,select,textarea{font:inherit;border:1px solid #abbeb8;border-radius:5px;padding:9px;background:white;color:inherit}
button{cursor:pointer}button:hover{background:#e3eee8}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #6e9ca6}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:16px 0}.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:24px}
article,aside{background:#fffefa;padding:24px;border:1px solid #d8e1d8;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}label{display:block;margin:12px 0 5px}input,textarea,aside select{width:100%}textarea{min-height:95px}.muted{font-size:14px;color:#607276}#message{min-height:25px}#raw{min-height:150px;font:12px monospace}#jump{max-width:210px}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:16px}}
</style><main><h1>Intended-answer review</h1>
<p>Read the response and transcribe its intended final answer. “Numeric” does not mean correct.
Use absent or ambiguous when no single final answer is identifiable. Do not solve the question to supply an answer.</p>
<p class="muted">For numeric answers, paste an exact supporting excerpt. Downloads may contain pending entries.
Save the draft through the CLI to create a durable revision, then import that exported revision before continuing.
Browser storage is only a convenience cache. No labels are confirmed by this page.</p>
<div class="bar"><button id="previous">Previous</button><button id="next">Next</button><button id="pending">Next pending</button><select id="jump" aria-label="Response"></select><strong id="progress" aria-live="polite"></strong></div>
<div class="bar"><button id="download">Download all edits</button><label for="file">Import saved draft</label><input id="file" type="file" accept=".json,application/json"></div>
<details><summary>Copyable backup if download is unavailable</summary><p>Use “Refresh backup”, then copy the full JSON to a local file. This includes pending and incomplete entries.</p><button id="backup">Refresh backup</button><textarea id="raw" aria-label="JSON backup" readonly></textarea></details>
<p id="message" role="status"></p><div id="workspace" class="grid"><article><div id="identity" class="muted"></div><h2>Question</h2><pre id="question"></pre><h2>Response</h2><pre id="response"></pre></article><aside>
<label for="reviewer">Reviewer</label><input id="reviewer" autocomplete="off">
<label for="status">Intended answer</label><select id="status"><option>pending</option><option>numeric</option><option>absent</option><option>ambiguous</option></select>
<label for="answer">Numeric transcription</label><input id="answer" placeholder="One number, no units" autocomplete="off">
<label for="evidence">Exact supporting response text</label><textarea id="evidence"></textarea>
<label for="note">Note</label><textarea id="note"></textarea></aside></div>
<script type="application/json" id="data">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),rows=data.rows;
const blank=r=>({response_id:r.response_id,status:'pending',answer:null,evidence:'',note:''});
let draft={packet_sha256:data.packet_sha256,base_revision:0,reviewer:'',labels:rows.map(blank)},index=0;
const cache='sampling-sft-eligibility:'+data.packet_sha256;
function message(s){$('message').textContent=s;}
function restore(value){
 if(value.packet_sha256!==data.packet_sha256||!Number.isInteger(value.base_revision)||value.base_revision<0||typeof value.reviewer!=='string'||!Array.isArray(value.labels)||value.labels.length!==rows.length)throw Error('Wrong packet or draft schema.');
 const map=new Map(value.labels.map(r=>[r.response_id,r]));
 if(map.size!==rows.length||rows.some(r=>!map.has(r.response_id)))throw Error('Unknown or duplicate response.');
 for(const r of value.labels){if(!['pending','numeric','absent','ambiguous'].includes(r.status)||typeof r.evidence!=='string'||typeof r.note!=='string'||(r.answer!==null&&typeof r.answer!=='string'))throw Error('Malformed label.');}
 draft={packet_sha256:value.packet_sha256,base_revision:value.base_revision,reviewer:value.reviewer,labels:rows.map(r=>map.get(r.response_id))};
}
function persist(){try{localStorage.setItem(cache,JSON.stringify(draft));}catch(e){message('Browser storage unavailable; download or copy a backup.');}}
function progress(){const n=draft.labels.filter(r=>r.status!=='pending'&&(r.status!=='numeric'||(r.answer||'').trim()&&r.evidence.trim())).length;$('progress').textContent=n+'/'+rows.length+' labelled · saved revision '+draft.base_revision;}
function render(){
 progress();$('reviewer').value=draft.reviewer;
 if(!rows.length){$('workspace').hidden=true;$('previous').disabled=true;$('next').disabled=true;$('pending').disabled=true;message('No required extraction-failure reviews. Coverage may be frozen through the CLI.');return;}
 const r=rows[index],l=draft.labels[index];$('identity').textContent=(index+1)+' / '+rows.length+' · '+r.response_id;
 $('question').textContent=r.question;$('response').textContent=r.response;
 for(const id of ['status','answer','evidence','note'])$(id).value=l[id]||'';
 $('answer').disabled=l.status!=='numeric';$('jump').value=String(index);
 $('previous').disabled=index===0;$('next').disabled=index===rows.length-1;
}
rows.forEach((r,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent='Response '+(i+1);$('jump').appendChild(o);});
$('reviewer').addEventListener('input',()=>{draft.reviewer=$('reviewer').value;persist();});
for(const id of ['status','answer','evidence','note'])$(id).addEventListener(id==='status'?'change':'input',()=>{if(!rows.length)return;const l=draft.labels[index];l[id]=$(id).value;if(id==='status'&&l.status!=='numeric')l.answer=null;persist();if(id==='status')render();else progress();});
$('previous').onclick=()=>{if(index>0){index--;render();}};$('next').onclick=()=>{if(index<rows.length-1){index++;render();}};
$('jump').onchange=()=>{index=Number($('jump').value);render();};
$('pending').onclick=()=>{for(let i=1;i<=rows.length;i++){const j=(index+i)%rows.length,r=draft.labels[j];if(r.status==='pending'||r.status==='numeric'&&(!(r.answer||'').trim()||!r.evidence.trim())){index=j;render();return;}}message('No pending entries. Download the draft and save it through the CLI.');};
$('backup').onclick=()=>{$('raw').value=JSON.stringify(draft,null,2)+'\\n';$('raw').select();message('Backup includes all edits, including pending entries.');};
$('download').onclick=()=>{const text=JSON.stringify(draft,null,2)+'\\n';$('raw').value=text;const blob=new Blob([text],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='sampling-sft-labels.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);message('Download requested. If unavailable, use the copyable backup.');};
$('file').onchange=async()=>{try{if(!$('file').files[0])return;const value=JSON.parse(await $('file').files[0].text());if(!confirm('Replace this page draft? Preserve your current edits first.'))return;restore(value);persist();render();message('Saved draft imported.');}catch(e){message(e.message);}};
try{const value=localStorage.getItem(cache);if(value)restore(JSON.parse(value));}catch(e){message('Cached draft unavailable: '+e.message);}
render();
</script></main></html>'''
