"""Paired readable-trace reductions; no model imports or readout exposure."""
import numpy as np


class CaptureFailure(ValueError):
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def summarize_pairs(batches, *, sites, content_tokens=128,
                    retained_sites=('block-14-output','block-17-output','block-18-output',
                                    'block-19-output','block-20-output','block-21-output')):
    """Reduce named-site batches using explicit absolute content-position maps.

    Backends supply base/adapted arrays [examples, input positions, width].
    Subtraction precedes retention conversion; this operation retains FP64
    evidence. Storage precision and runtime qualification are separate gates.
    """
    sites=list(sites);retained_sites=list(retained_sites)
    if len(sites)!=len(set(sites)) or not set(retained_sites)<=set(sites):
        raise ValueError('Named sites must be unique and cover retained sites.')
    if content_tokens<5:raise ValueError('Primary window requires positions 0–4.')
    sums={};primary={s:[] for s in sites};baseline={s:[] for s in sites}
    retained={s:[] for s in retained_sites};ids=[];moments=None
    for batch in batches:
        names=batch['example_ids'];maps=batch['content_positions']
        if len(names)!=len(maps) or len(set(names))!=len(names) or set(ids)&set(names):
            raise ValueError('Unique example IDs and matching position maps required.')
        if set(batch['base'])!=set(sites) or set(batch['adapted'])!=set(sites):
            raise ValueError('Capture sites differ from the declared site inventory.')
        for site in sites:
            base=np.asarray(batch['base'][site],dtype=np.float64)
            adapted=np.asarray(batch['adapted'][site],dtype=np.float64)
            if base.ndim!=3 or base.shape!=adapted.shape or base.shape[0]!=len(names):
                raise ValueError('Paired activation shapes differ.')
            if not np.isfinite(base).all() or not np.isfinite(adapted).all():
                raise ValueError('Nonfinite captured activation.')
            if site not in sums:sums[site]=np.zeros((content_tokens,base.shape[2]),dtype=np.float64)
            if sums[site].shape!=(content_tokens,base.shape[2]):raise ValueError('Hidden width changed.')
            for row,positions in enumerate(maps):
                positions=np.asarray(positions)
                if (positions.shape!=(content_tokens,) or not np.issubdtype(positions.dtype,np.integer)
                        or np.any(positions<0) or np.any(positions>=base.shape[1])
                        or np.any(np.diff(positions)<=0)):
                    raise ValueError('Content position map must be ordered unique integer input positions.')
                b=base[row,positions];delta=adapted[row,positions]-b
                sums[site]+=delta
                primary[site].append(delta[:5].mean(axis=0))
                baseline[site].append(b[:5].mean(axis=0))
                if site in retained:retained[site].append(delta)
                if site=='block-14-output':
                    h=b[1:]
                    if moments is None:moments=dict(count=0,sum=np.zeros(b.shape[1]),outer_sum=np.zeros((b.shape[1],b.shape[1])))
                    moments['count']+=len(h);moments['sum']+=h.sum(axis=0);moments['outer_sum']+=h.T@h
        ids.extend(names)
    if not ids:raise ValueError('Capture stream is empty.')
    return dict(example_ids=ids,position_delta_sums=sums,
                example_primary_delta={s:np.stack(v) for s,v in primary.items()},
                example_primary_base={s:np.stack(v) for s,v in baseline.items()},
                retained_delta={s:np.stack(v) for s,v in retained.items()},base_moments=moments)


def logit_lens(vector, *, norm_weight, unembedding, epsilon):
    """Apply the pinned model's RMS norm gain and unembedding, without bias.

    Numeric output only: callers own the qualified source and reveal gate.
    A zero difference has no interpretable readout and returns None.
    """
    v=np.asarray(vector,dtype=np.float64);gain=np.asarray(norm_weight,dtype=np.float64)
    matrix=np.asarray(unembedding,dtype=np.float64)
    if v.ndim!=1 or gain.shape!=v.shape or matrix.ndim!=2 or matrix.shape[1]!=len(v):
        raise ValueError('Logit-lens shape mismatch.')
    if not all(np.isfinite(x).all() for x in (v,gain,matrix)) or not np.isfinite(epsilon) or epsilon<0:
        raise ValueError('Nonfinite logit-lens input or invalid norm epsilon.')
    if not np.any(v):return None
    return matrix@(gain*v/np.sqrt(np.mean(v*v)+epsilon))


def retention_evidence(values, *, storage_dtype):
    """Report measured storage error; no numerical allowance or pass is inferred."""
    source=np.asarray(values)
    if not source.size or not np.isfinite(source).all():raise ValueError('Nonfinite or empty retained values.')
    dtype=np.dtype(storage_dtype)
    if dtype not in (np.dtype('float16'),np.dtype('float32'),np.dtype('float64')):
        raise ValueError('Retained storage must be floating point.')
    stored=source.astype(dtype)
    if not np.isfinite(stored).all():raise ValueError('Nonfinite retained conversion.')
    error=np.abs(stored.astype(np.float64)-source.astype(np.float64))
    coordinate=list(np.unravel_index(int(np.argmax(error)),error.shape))
    return dict(qualified=False,source_dtype=str(source.dtype),storage_dtype=str(dtype),
                max_absolute_error=float(error.max()),worst_coordinate=[int(x) for x in coordinate],
                source_value=float(source[tuple(coordinate)]),stored_value=float(stored[tuple(coordinate)]),
                error_l2=float(np.linalg.norm(error.ravel())))


class HookedTraceCapture:
    """Capture embedding and decoder-block outputs in a single model call.

    Accepts the actual embedding module and ordered decoder blocks. Works with
    Torch modules or independent CPU hook fixtures; the caller owns inference
    mode, adapter selection and input construction.
    """
    def __init__(self, embedding, blocks):
        self.modules={'embedding-output':embedding}
        self.modules.update({f'block-{i}-output':block for i,block in enumerate(blocks)})
        if len({id(m) for m in self.modules.values()})!=len(self.modules):
            raise ValueError('Named capture modules must be distinct.')

    def forward(self, call):
        outputs={};handles=[]
        def save(name,output):
            if name in outputs:raise ValueError(f'Capture site fired twice: {name}')
            value=output[0] if isinstance(output,tuple) else output
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            value=np.array(value,copy=True)
            if value.ndim!=3:
                raise CaptureFailure(f'Invalid activation shape at {name}',dict(site=name,shape=list(value.shape)))
            if not np.isfinite(value).all():
                coordinate=np.argwhere(~np.isfinite(value))[0].tolist()
                raise CaptureFailure(f'Nonfinite activation at {name}',dict(site=name,coordinate=coordinate,
                    observed=repr(float(value[tuple(coordinate)])),allowance='finite values required'))
            outputs[name]=value
        try:
            for name,module in self.modules.items():
                handles.append(module.register_forward_hook(lambda m,a,o,name=name:save(name,o)))
            call()
            if set(outputs)!=set(self.modules):raise ValueError('Missing named capture sites.')
            return outputs
        finally:
            for handle in handles:handle.remove()
