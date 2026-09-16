"""A single, explicit voxel domain for surface display, measurements and OpenFOAM.

No surface smoothing is used for the CFD mesh. Source screenshots and original
segmentations are never overwritten. All grid/segmentation edits are recorded.
Source arrays are slice Z, image Y, image X; solver arrays and geometry
coordinates are world X,Y,Z in millimetres.
"""
from __future__ import annotations
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import trimesh
import config as C

ROOT = C.WORK_DIR / 'cfd'
VERSION = 'voxel-domain-5-xyz'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def world_order(a):
    # Input is [slice(Z), image-Y, image-X]. Viewer/CFD arrays are
    # [world-X, world-Y, world-Z]: X is image X, Y is reversed slice index,
    # and Z is reversed image-Y.
    return np.ascontiguousarray(a[::-1, ::-1, :].transpose(2, 0, 1))


def prepare_source():
    """Cache a cropped, calibrated source and explicit exterior-contact labels."""
    ROOT.mkdir(parents=True, exist_ok=True)
    paths = [C.WORK_DIR / 'airway_fine.npy', C.WORK_DIR / 'volume_meta.json',
             C.WORK_DIR / 'vol_fine.npy']
    identity = digest(VERSION.encode() + b''.join(hashlib.sha256(p.read_bytes()).digest() for p in paths))
    target = ROOT / 'source.npz'
    if target.exists():
        with np.load(target) as z:
            if str(z['source_hash']) == identity:
                return target
    from airway import external_air
    meta = json.loads(paths[1].read_text())
    a = np.load(paths[0]); v = np.load(paths[2])
    sp = np.asarray(meta['grids']['fine']['spacing_mm'])
    fov = meta['fov_circle_px']; ds = meta['grids']['fine']['downsample']
    y, x = np.mgrid[:a.shape[1], :a.shape[2]]
    disc = np.hypot((x+.5)*ds-fov['cx'], (y+.5)*ds-fov['cy']) < fov['r']-1.5*ds
    ext = external_air(v, a, disc, (.3, .7, .7))
    contact = a & ndi.binary_dilation(ext)
    labels, _ = ndi.label(ndi.binary_dilation(contact, iterations=2) & a)
    counts = np.bincount(labels[contact]); counts[0] = 0
    ids = np.argsort(counts)[::-1][:2]
    if len(ids) != 2 or min(counts[ids]) < 20:
        raise ValueError('Cannot identify two substantial nostril contacts; review segmentation')
    ids = sorted(ids, key=lambda k: np.nonzero(contact & (labels == k))[2].mean())
    # Names are anatomical: positive world X is the patient's left.
    ports = np.zeros(a.shape, np.uint8)
    for pid, k in zip([2, 1], ids):
        ports[contact & (labels == k)] = pid
    unexpected = int((contact & (ports == 0)).sum())
    if unexpected:
        raise ValueError(f'{unexpected} exterior contact voxels outside the two nostrils; review before solving')
    ports[-1][a[-1]] = 3
    full_shape = a.shape
    a, ext, ports, v = map(world_order, (a, ext, ports, v))
    spacing = sp[[2, 0, 1]]  # array order world X, world Y, world Z
    centre = np.array([C.SRC_W*meta['px_mm']/2, (full_shape[0]-1)*sp[0]/2,
                       C.SRC_H*meta['px_mm']/2])
    origin = np.array([-centre[0], -centre[1]-sp[0]/2,
                       centre[2]-full_shape[1]*sp[1]])
    coords = np.argwhere(a)
    lo = np.maximum(coords.min(axis=0)-12, 0)
    hi = np.minimum(coords.max(axis=0)+13, a.shape)
    sl = tuple(slice(i, j) for i, j in zip(lo, hi))
    origin += lo*spacing
    np.savez_compressed(target, mask=a[sl], exterior=ext[sl], ports=ports[sl], gray=v[sl],
                        spacing=spacing, origin=origin, source_hash=identity,
                        source_shape=full_shape)
    return target


def settings_displacement(points, settings, profiles):
    """Prescribed local wall offset, in mm. It is NOT a tissue mechanics solve."""
    out = np.zeros(len(points))
    for side in ['L', 'R']:
        p = profiles['sides'][side]['profile']
        centres = np.asarray(p['centers'])
        distance, idx = cKDTree(centres).query(points)
        t = np.asarray(p['s_mm'])[idx] / profiles['sides'][side]['length_mm']
        selected = (points[:, 0] > 0) if side == 'L' else (points[:, 0] <= 0)
        selected &= points[:, 2] > profiles['choana_world_z']
        selected &= distance < 15
        weights = {}
        for name, (centre, width) in {'head':(.38,.16), 'body':(.64,.22), 'valve':(.17,.12)}.items():
            u = np.abs((t-centre)/width)
            weights[name] = np.where(u<1, (1-u*u)**2, 0)
        position = settings.get('position', 'supine')
        dependent = position == ('left' if side == 'L' else 'right')
        opposite = position == ('right' if side == 'L' else 'left')
        amplitude = settings.get('response'+side, .6)
        vascular = amplitude if dependent else -.25*amplitude if opposite else -.3*amplitude if position=='upright' else 0
        settled = 1-np.exp(-settings.get('elapsed',10)/max(.1,settings.get('tau',5)))
        swelling = (vascular+settings.get('cycle',0)*(1 if side=='L' else -1))*settled
        swelling += settings.get('gravity',.15)*(1 if dependent else -1 if opposite else 0)
        change = sum(settings.get('operations',{}).get(side,{}).get(k,0)*w for k,w in weights.items())
        change -= swelling*np.maximum(weights['head'],weights['body'])*(1-settings.get('relief',0)/100 if swelling>0 else 1)
        # Widen the lateral wall only. Isotropic dilation would also erode the
        # septum and could create a non-anatomical left/right shortcut.
        lateral=(points[:,0]>centres[idx,0]) if side=='L' else (points[:,0]<centres[idx,0])
        change=np.where((change>0)&~lateral,0,change)
        out[selected] = change[selected]
    return out


def repair_edge_contacts(mask, score):
    """Remove the weaker voxel at an ambiguous diagonal; never bridge a septum.

    Counts and disconnected removals are reported. A blocked nasal branch fails
    port validation instead of being silently reconnected.
    """
    result = mask.copy(); removed = 0
    for _ in range(12):
        kill = np.zeros_like(result)
        for ax in range(3):
            m = np.moveaxis(result, ax, 0); s = np.moveaxis(score, ax, 0)
            k = np.moveaxis(kill, ax, 0)
            a,b,c,d = m[:,:-1,:-1],m[:,1:,:-1],m[:,:-1,1:],m[:,1:,1:]
            for first, second, i, j in [(a&d&~b&~c, s[:,:-1,:-1]<=s[:,1:,1:],(slice(None),slice(None,-1),slice(None,-1)),(slice(None),slice(1,None),slice(1,None))),
                                        (b&c&~a&~d, s[:,1:,:-1]<=s[:,:-1,1:],(slice(None),slice(1,None),slice(None,-1)),(slice(None),slice(None,-1),slice(1,None)))]:
                k[i] |= first & second; k[j] |= first & ~second
        count = int((result & kill).sum())
        if not count: break
        result[kill] = False; removed += count
    return result, removed


def regularize(mask,score):
    # A point shared by two otherwise disconnected surface fans is not a
    # manifold vertex, even when every edge has exactly two incident faces.
    bad = np.zeros(256,bool)
    for bits in range(256):
        cube=np.array([(bits>>k)&1 for k in range(8)],bool).reshape(2,2,2)
        bad[bits]=ndi.label(cube)[1]>1 or ndi.label(~cube)[1]>1
    result=mask.copy(); diagonal=thin=corner=0
    for _ in range(30):
        before=int(result.sum())
        result,n=repair_edge_contacts(result,score);diagonal+=n
        pattern=np.zeros(np.array(result.shape)-1,np.uint8)
        offsets=list(np.ndindex(2,2,2))
        for bit,offset in enumerate(offsets):
            sl=tuple(slice(o,o+pattern.shape[k]) for k,o in enumerate(offset))
            pattern |= result[sl].astype('u1')<<bit
        locations=np.argwhere(bad[pattern])
        if len(locations):
            cells=locations[:,None,:]+np.array(offsets)[None,:,:]
            values=score[tuple(cells.transpose(2,0,1))].copy()
            values[~result[tuple(cells.transpose(2,0,1))]]=np.inf
            chosen=cells[np.arange(len(cells)),values.argmin(axis=1)]
            count=int(result.sum());result[tuple(chosen.T)]=False;corner+=count-int(result.sum())
        if int(result.sum())==before: break
    else: raise ValueError('Domain regularization did not converge')
    return result,dict(diagonalVoxelsRemoved=diagonal,cornerVoxelsRemoved=corner)


def flatten_nostrils(domain, ports, exterior):
    """Give each nostril one planar opening (in place); returns per-nostril audit.

    The segmented nostril rim is an oblique stair-step surface facing down,
    forward and sideways. Its open faces are extruded outward through exterior
    air, along the nostril's dominant opening axis, to the plane of its most
    outward face, so the opening becomes a single flat face on that plane.
    The added cells are exterior air, walled virtually on their sides; the
    airway itself is not edited. Exterior labels are then kept only on that
    plane's outer layer, so no other face can open. Used by the lattice
    solver, whose ghost-cell pressure openings misbehave on stair corners.
    """
    directions = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    padded_domain = np.pad(domain,1); padded_exterior = np.pad(exterior,1)
    keep_exterior = np.zeros_like(exterior); out = {}
    for label, name in [(1,'left_nostril'),(2,'right_nostril')]:
        owners = np.argwhere(domain & (ports==label))
        if not len(owners): raise ValueError(f'{name} has no cells')
        counts = np.zeros(6,int); faces = []
        for k, delta in enumerate(directions):
            nb = owners+1+delta
            open_face = ~padded_domain[tuple(nb.T)] & padded_exterior[tuple(nb.T)]
            counts[k] = open_face.sum(); faces.append(owners[open_face])
        if not counts.any(): raise ValueError(f'{name} has no open faces')
        k = int(counts.argmax()); axis, step = k//2, 1 if k%2==0 else -1
        # Every owner of an open face reaches the plane of the most outward open face.
        all_owners = np.unique(np.concatenate([f for f in faces if len(f)]),axis=0)
        level = all_owners[:,axis].max() if step>0 else all_owners[:,axis].min()
        added = 0; blocked = 0
        for cell in all_owners:
            position = cell.copy()
            while position[axis] != level:
                position[axis] += step
                if domain[tuple(position)]: continue
                if not exterior[tuple(position)]: blocked += 1; break  # tissue: this column stays stepped
                domain[tuple(position)] = True; ports[tuple(position)] = label; added += 1
        plane = level+step
        if not (0 <= plane < domain.shape[axis]): raise ValueError(f'{name} opening plane leaves the domain box')
        # Only this nostril's footprint on its plane may stay open.
        layer = [slice(lo,hi+1) for lo,hi in zip(all_owners.min(0),all_owners.max(0))]; layer[axis] = plane
        keep_exterior[tuple(layer)] |= exterior[tuple(layer)]
        out[name] = dict(axis='xyz'[axis], direction=int(step), planeIndex=int(plane), cellsAdded=int(added), columnsBlockedByTissue=int(blocked),
                         openFacesBefore={('+' if d%2==0 else '-')+'xyz'[d//2]:int(c) for d,c in enumerate(counts) if c})
    exterior &= keep_exterior
    return out


def connect_missing_ports(domain, walkable, ports):
    """Restore a 6-connected path when voxel swelling pinches a still-patent nostril.

    The 1-D tube model can keep a millimetre-scale lumen that a 0.7 mm stair-step
    plus regularization splits. Walking the pre-swelling lumen joins the nostril
    without reopening the rest of the prescribed wall motion.
    """
    restored = 0
    neigh = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    nz, ny, nx = domain.shape
    for pid in (1, 2):
        if (domain & (ports == pid)).any():
            continue
        seeds = [tuple(p) for p in np.argwhere(walkable & (ports == pid))]
        if not seeds:
            continue
        prev = {}; seen = set(seeds); queue = deque(seeds); hit = None
        while queue:
            x = queue.popleft()
            if domain[x]:
                hit = x; break
            i, j, k = x
            for di, dj, dk in neigh:
                y = (i+di, j+dj, k+dk)
                if y in seen or not (0 <= y[0] < nz and 0 <= y[1] < ny and 0 <= y[2] < nx):
                    continue
                if not walkable[y]:
                    continue
                seen.add(y); prev[y] = x; queue.append(y)
        if hit is None:
            continue
        cur = prev.get(hit)
        while cur is not None:
            if not domain[cur]:
                domain[cur] = True; restored += 1
            cur = prev.get(cur)
    return restored


def keep_opening_components(repaired, ports):
    """Do not drop a nostril or outlet that regularization split off the largest body."""
    lab, n = ndi.label(repaired)
    counts = np.bincount(lab.ravel()); counts[0] = 0
    if not counts.any():
        raise ValueError('No patent airway remains')
    keep = {int(counts.argmax())}
    for pid in (1, 2, 3):
        for i in np.unique(lab[ports == pid]):
            if i:
                keep.add(int(i))
    domain = np.isin(lab, list(keep))
    return domain, int(repaired.sum() - domain.sum())


def repair_collar_manifold(domain, added, score):
    """Nostril-collar extrusion can leave edge-adjacent voxels; keep the collar."""
    if not added.any():
        return domain, 0
    pref = np.where(domain, np.maximum(score, 1.0), -1.0)
    pref[added] = float(np.max(pref)) + 10
    domain, n = repair_edge_contacts(domain, pref)
    domain, extra = regularize(domain, pref)
    return domain, n + extra.get('diagonalVoxelsRemoved', 0) + extra.get('cornerVoxelsRemoved', 0)


def build_domain(settings=None, spacing_mm=.7, threshold_offset=0, subdivisions=2, flat_nostrils=False):
    source_path = prepare_source()
    with np.load(source_path) as src:
        src = {k:src[k] for k in src.files}
    mask, sp, origin = src['mask'], src['spacing'], src['origin']
    # Positive distance in lumen; zero at the source voxel interface.
    inside = ndi.distance_transform_edt(mask, sampling=sp)
    outside = ndi.distance_transform_edt(~mask, sampling=sp)
    signed = np.where(mask, inside-.5*sp.min(), -(outside-.5*sp.min()))
    shape = np.ceil(np.array(mask.shape)*sp/spacing_mm).astype(int)
    coords = np.indices(shape, dtype=np.float32).reshape(3,-1)
    points = (coords.T+.5)*spacing_mm+origin
    sample = ((coords+.5)*spacing_mm/sp[:,None]-.5)
    score = ndi.map_coordinates(signed,sample,order=1,mode='constant',cval=-100)
    if settings:
        profiles = json.loads((C.DATA_DIR/'pre/airway.json').read_text())
        offset = settings_displacement(points,settings,profiles)
        gray=ndi.map_coordinates(src['gray'].astype(np.float32),sample,order=1,mode='nearest')
        offset[(offset>0)&(gray>=96)]=0  # preserve visible bone; no unsegmented bone cut
        # Preserve inlet/outlet collars; region operations must not edit ports.
        port_dist = ndi.distance_transform_edt(src['ports']==0,sampling=sp)
        distance = ndi.map_coordinates(port_dist,sample,order=1,mode='nearest')
        offset *= np.clip((distance-3)/3,0,1)
        score += offset
    score += threshold_offset
    score = score.reshape(shape)
    candidate = score > 0
    base_lumen = candidate if not settings else ((score.reshape(-1) - offset) > 0).reshape(shape)
    repaired, repairs = regularize(candidate,score)
    port_distance, indices = ndi.distance_transform_edt(src['ports']==0,sampling=sp,return_indices=True)
    nearest = src['ports'][tuple(indices)]
    d = ndi.map_coordinates(port_distance,sample,order=1,mode='nearest').reshape(shape)
    ports = ndi.map_coordinates(nearest,sample,order=0,mode='nearest').reshape(shape)
    exterior = ndi.map_coordinates(src['exterior'].astype(np.float32),sample,order=1,mode='constant',cval=0).reshape(shape) > .25
    ports[d > max(spacing_mm*1.6,.8)] = 0
    # Outlet lies on inferior scan cut; no artificial connection to exterior.
    ports[(np.indices(shape)[1] > 1) & (ports==3)] = 0
    domain, discarded = keep_opening_components(repaired, ports)
    restored = connect_missing_ports(domain, base_lumen | domain, ports)
    collar = None
    if flat_nostrils:
        before = domain.copy()
        collar = flatten_nostrils(domain,ports,exterior)
        domain, collar_repair = repair_collar_manifold(domain, domain & ~before, score)
        repairs = dict(repairs, collarManifoldVoxelsRemoved=collar_repair)
        domain, dropped = keep_opening_components(domain, ports)
        discarded += dropped
        restored += connect_missing_ports(domain, base_lumen | domain, ports)
    ident = digest(VERSION.encode()+domain.tobytes()+ports.tobytes()+exterior.tobytes()+np.asarray([*origin,spacing_mm],dtype='<f8').tobytes())
    audit = dict(version=VERSION,sourceHash=str(src['source_hash']),geometryHash=ident,
                 source='CT PNG screenshots; supine',clinicalReview=False,
                 spacingMm=spacing_mm,cells=int(domain.sum()),volumeCc=float(domain.sum()*spacing_mm**3/1000),
                 sourceVolumeCc=float(mask.sum()*np.prod(sp)/1000),
                 **repairs,disconnectedVoxelsRemoved=discarded,splitPassageVoxelsRestored=int(restored),
                 prescribedWallMotion=bool(settings and np.any(offset)),
                 sourceExteriorContacts=2,expectedOpenings=['left_nostril','right_nostril','outlet'],
                 uncertainty='Screenshot window clipping, inferred in-plane scale, segmentation and voxel-wall discretization. No tissue mechanics or clinical validation.')
    if collar: audit.update(flatNostrils=True,nostrilCollars=collar,
                            nostrilCollarNote='each nostril is extended through exterior air with a short collar of fluid cells to a single plane normal to its dominant opening direction; the collar walls are virtual')
    if subdivisions not in [1,2,3]: raise ValueError('Invalid cell subdivision')
    if subdivisions>1:
        # Resolve one-voxel-wide passages without changing or deleting anatomy.
        # This is a solver-grid refinement of the SAME stair-step boundary.
        for axis in range(3):
            domain=np.repeat(domain,subdivisions,axis=axis)
            ports=np.repeat(ports,subdivisions,axis=axis)
            exterior=np.repeat(exterior,subdivisions,axis=axis)
    audit.update(reconstructionSpacingMm=spacing_mm,spacingMm=spacing_mm/subdivisions,
                 subdivisions=subdivisions,cells=int(domain.sum()))
    return dict(mask=domain,ports=ports,exterior=exterior,origin=origin,h=spacing_mm/subdivisions,audit=audit)


OFFSETS = [
    [[1,0,0],[1,1,0],[1,1,1],[1,0,1]], # +X
    [[0,0,0],[0,0,1],[0,1,1],[0,1,0]], # -X
    [[0,1,0],[0,1,1],[1,1,1],[1,1,0]], # +Y
    [[0,0,0],[1,0,0],[1,0,1],[0,0,1]], # -Y
    [[0,0,1],[1,0,1],[1,1,1],[0,1,1]], # +Z
    [[0,0,0],[0,1,0],[1,1,0],[1,0,0]], # -Z
]


def mesh_arrays(domain):
    mask, ports = domain['mask'],domain['ports']
    xyz = np.argwhere(mask)
    idx = np.full(mask.shape,-1,dtype=np.int32); idx[mask] = np.arange(len(xyz))
    padded = np.pad(idx,1,constant_values=-1)
    outside = np.pad(domain.get('exterior',np.ones_like(mask)),1,constant_values=False)
    internal=[]; boundary=[]
    for direction, delta in enumerate([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]):
        nb = padded[tuple((xyz+1+delta).T)]
        owner = np.arange(len(xyz)); face = xyz[:,None,:]+np.array(OFFSETS[direction])[None,:,:]
        keep = nb > owner
        internal.append((face[keep],owner[keep],nb[keep]))
        keep = nb < 0
        p = ports[tuple(xyz[keep].T)].copy()
        # An inlet face must actually face exterior air. A nearby nasal wall
        # remains no-slip; proximity alone must never become an opening.
        faces_outside = outside[tuple((xyz[keep]+1+delta).T)]
        p[((p==1)|(p==2)) & ~faces_outside] = 0
        # Outlet only on faces pointing inferior, on the actual cut.
        if direction != 3: p[p==3]=0
        boundary.append((face[keep],owner[keep],p))
    fi,oi,ni = (np.concatenate([t[j] for t in internal]) for j in range(3))
    order = np.lexsort((ni,oi)); fi,oi,ni = fi[order],oi[order],ni[order]
    fb,ob,pb = (np.concatenate([t[j] for t in boundary]) for j in range(3))
    order = np.argsort(pb,kind='stable'); fb,ob,pb = fb[order],ob[order],pb[order]
    faces = np.concatenate([fi,fb]); owner=np.r_[oi,ob]
    points, inverse = np.unique(faces.reshape(-1,3),axis=0,return_inverse=True)
    faces = inverse.reshape(-1,4)
    patches=[]; start=len(fi)
    for pid,name in enumerate(['walls','left_nostril','right_nostril','outlet']):
        n=int((pb==pid).sum())
        if not n: raise ValueError(f'{name} has no faces: airway may be blocked at this resolution')
        patches.append(dict(name=name,type='wall' if pid==0 else 'patch',startFace=start,nFaces=n,areaMm2=n*domain['h']**2))
        start+=n
    world = points*domain['h']+domain['origin']
    centres = (xyz+.5)*domain['h']+domain['origin']
    surface = faces[len(fi):]
    triangles = np.concatenate([surface[:,[0,1,2]],surface[:,[0,2,3]]])
    m=trimesh.Trimesh(world,triangles,process=True)
    domain['audit'].update(watertight=bool(m.is_watertight),windingConsistent=bool(m.is_winding_consistent),
                           surfaceVolumeCc=float(m.volume/1000),boundaryFaces=len(fb),patches=patches)
    if not m.is_watertight or not m.is_winding_consistent:
        raise ValueError('CFD surface is not a closed oriented manifold; refusing solve')
    return dict(points=world/1000,faces=faces,owner=owner,neighbour=ni,patches=patches,
                cells=xyz,centres=centres,surface=m,
                boundaryOwner=ob,boundaryPatch=pb)


def foam_header(name,cls='dictionary'):
    return f'FoamFile\n{{ version 2.0; format ascii; class {cls}; object {name}; }}\n'


def write_poly_mesh(domain, arrays, case):
    folder=case/'constant/polyMesh'; folder.mkdir(parents=True,exist_ok=True)
    def write(name,cls,data,fmt):
        with (folder/name).open('w') as f:
            f.write(foam_header(name,cls)+f'{len(data)}\n(\n')
            np.savetxt(f,data,fmt=fmt)
            f.write(')\n')
    write('points','vectorField',arrays['points'],'(%.10g %.10g %.10g)')
    write('faces','faceList',arrays['faces'],'4(%d %d %d %d)')
    write('owner','labelList',arrays['owner'],'%d')
    write('neighbour','labelList',arrays['neighbour'],'%d')
    with (folder/'boundary').open('w') as f:
        f.write(foam_header('boundary','polyBoundaryMesh')+'4\n(\n')
        for p in arrays['patches']:
            f.write('{name}\n{{ type {type}; nFaces {nFaces}; startFace {startFace}; }}\n'.format(**p))
        f.write(')\n')
    np.savez_compressed(case/'domain.npz',mask=domain['mask'],ports=domain['ports'],origin=domain['origin'],h=domain['h'],
                        centres=arrays['centres'],boundaryOwner=arrays['boundaryOwner'],boundaryPatch=arrays['boundaryPatch'])
    (case/'geometry.json').write_text(json.dumps(domain['audit'],indent=2))
    arrays['surface'].export(case/'airway.glb')
