"""Offline review-page interactions without model or network dependencies."""
import json
import re
import shutil
import subprocess

import pytest

from pilot_eval.sampling_sft_review_ui import page


def test_embedded_packet_has_no_executable_response_markup():
    html=page([dict(response_id='x',question='question',response='</script><script>alert(1)</script>')],'packet')
    encoded=re.search(r'id="data">(.*?)</script>',html,re.S).group(1)
    assert '<script>' not in encoded and json.loads(encoded)['rows'][0]['response'].startswith('</script>')
    assert 'Download all edits' in html and 'Copyable backup' in html


@pytest.mark.parametrize('rows',[[],[dict(response_id='a',question='q',response='The result is 12.')]])
def test_page_backup_pending_entries_and_saved_revision_import(tmp_path,rows):
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable; browser interaction fixture requires JavaScript runtime.')
    html=page(rows,'packet');data=re.search(r'id="data">(.*?)</script>',html,re.S).group(1)
    script=re.findall(r'<script>(.*?)</script>',html,re.S)[0]
    harness='''
const assert=require('node:assert/strict'), vm=require('node:vm');
const elements=new Map(), events={},downloads=[];
function element(id){if(!elements.has(id))elements.set(id,{value:'',textContent:'',files:[],disabled:false,hidden:false,
 addEventListener(type,fn){events[id+':'+type]=fn;},appendChild(){},remove(){},select(){},click(){if(this.download)downloads.push(this.download);}});return elements.get(id);}
element('data').textContent=DATA;
const context={document:{getElementById:element,createElement:()=>element('temporary'),body:{appendChild(){}}},
 localStorage:{getItem:()=>null,setItem(){}},Blob,URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}},
 setTimeout:fn=>fn(),confirm:()=>true};
vm.createContext(context);vm.runInContext(SCRIPT,context);
(async()=>{
element('backup').onclick();let initial=JSON.parse(element('raw').value);
assert.equal(initial.base_revision,0);assert.equal(initial.labels.length,COUNT);
assert(initial.labels.every(r=>r.status==='pending'));
element('download').onclick();assert.equal(downloads.length,1);
if(COUNT){
 element('status').value='numeric';events['status:change']();
 element('answer').value='12';events['answer:input']();
 element('evidence').value='result is 12';events['evidence:input']();
 element('backup').onclick();const edited=JSON.parse(element('raw').value);
 assert.equal(edited.labels[0].answer,'12');
 edited.base_revision=3;edited.reviewer='human';
 element('file').files=[{text:async()=>JSON.stringify(edited)}];await element('file').onchange();
 element('backup').onclick();assert.equal(JSON.parse(element('raw').value).base_revision,3);
}else assert.equal(element('workspace').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    harness=harness.replace('DATA',json.dumps(data)).replace('SCRIPT',json.dumps(script)).replace('COUNT',str(len(rows)))
    result=subprocess.run([node,'-e',harness],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
