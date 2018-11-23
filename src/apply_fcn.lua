-------------------------------- MNI Header -----------------------------------
-- @NAME       :  apply_fcn.lua
-- @DESCRIPTION:  Apply fully-convolutional network on a 3D volume
-- @COPYRIGHT  :
--               Copyright 2017 Vladimir Fonov, McConnell Brain Imaging Centre, 
--               Montreal Neurological Institute, McGill University.
--
--     This program is free software: you can redistribute it and/or modify
--     it under the terms of the GNU General Public License as published by
--     the Free Software Foundation, either version 3 of the License, or
--     (at your option) any later version.
-- 
--     This program is distributed in the hope that it will be useful,
--     but WITHOUT ANY WARRANTY; without even the implied warranty of
--     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--     GNU General Public License for more details.
-- 
--     You should have received a copy of the GNU General Public License
--     along with this program.  If not, see <http://www.gnu.org/licenses/>.
--             
------------------------------------------------------------------------

require 'torch'
require 'minc2_simple'
paths.dofile('tiles.lua')

local math=require 'math'

local function get_tile(minibatch, volumes, border, fov, corner, features)
    local corner=corner or {1,1,1}
    local sz=volumes[1]:size()
    local pidx={{corner[1],corner[1]+fov-1},{corner[2],corner[2]+fov-1},{corner[3],corner[3]+fov-1}}
    local adj_fov=fov-border*2
    local b_pidx={{corner[1]+border,corner[1]+fov-border-1},{corner[2]+border,corner[2]+fov-border-1},{corner[3]+border,corner[3]+fov-border-1}}
    
    local f
    for f=1,features do
        minibatch[ { 1, f, {},{},{} } ]:copy(volumes[f][pidx])
    end
end


local function put_tile_max(out_vol, tile, classes, border, fov, corner)
    local _,m
    
    local adj_fov  = fov-border-1
    local adj_fov2 = fov-border*2
    
    local _,m=tile:float():view(adj_fov2, adj_fov2, adj_fov2, classes):max(4)
    
    out_vol[ {{corner[1]+border,corner[1]+adj_fov},
              {corner[2]+border,corner[2]+adj_fov},
              {corner[3]+border,corner[3]+adj_fov}} ] = m

end

local function put_tile_feature(out_vol, tile, classes, border, fov, corner,feature)
    local _,m
    
    local adj_fov  = fov-border-1
    local adj_fov2 = fov-border*2
    
    m=tile:float():view(adj_fov2, adj_fov2, adj_fov2, classes)[{{},{},{},feature}]
    
    out_vol[ {{corner[1]+border,corner[1]+adj_fov},
              {corner[2]+border,corner[2]+adj_fov},
              {corner[3]+border,corner[3]+adj_fov}} ] = m

end

local function dump_tile_feature(out_file, tile, classes, border, fov, feature)
    local _,m
    
    local adj_fov  = fov-border-1
    local adj_fov2 = fov-border*2
    
    m=tile:float():view(adj_fov2, adj_fov2, adj_fov2, classes)[{{},{},{},feature}]
    
    dump_volume(out_file,nil, m)

end



function allocate_tile_minibatch_gpu(samples, border, fov, features)
    local volume_sz=torch.LongTensor(3)
    volume_sz:fill(fov)
    return torch.CudaTensor( samples, features, volume_sz[1] ,volume_sz[2], volume_sz[3] )
end

function allocate_tile_minibatch_cpu(samples, border, fov, features)
    local volume_sz=torch.LongTensor(3)
    volume_sz:fill(fov)
    return torch.Tensor( samples, features, volume_sz[1] ,volume_sz[2], volume_sz[3] )
end


local function allocate_tile_minibatch(samples, border, fov, features)
    return allocate_tile_minibatch_gpu(samples, border, fov, feaures)
end

function apply_fcn( opt )
    local in_volumes  = opt.in_volumes
    local in_files    = opt.in_files --TODO
    local out_volume  = opt.out_volume
    local out_raw     = opt.out_raw
    
    local features    = opt.features
    local outputs     = opt.outputs or 1
    local discrete    = opt.discrete 
    local model       = opt.model
    local fov         = opt.fov
    local border      = opt.border
    local classes     = opt.classes
    local default_out = opt.default_out
    local shift       = opt.shift
    
    if shift == nil then
        shift=0
    end
    
    if default_out==nil then
        default_out=0
    end
    
    if discrete==nil then
        discrete=true
    end
    
    local _minibatch=opt.minibatch
    
    local dsize     = in_volumes[1]:size() -- TODO: load from file if needed
    local dsize_out = out_volume:size()
    
--     assert(torch.all(torch.eq(
--              torch.LongTensor(dsize:size()    ,1, torch.LongStorage({2})),
--              torch.LongTensor(dsize_out:size(),1, torch.LongStorage({2}))
--           ) ) )
-- TODO: check dimensions of all features
    
    local adj_fov=fov-border*2
    
    if _minibatch==nil then
        _minibatch = allocate_tile_minibatch(1, border, fov, features)
    end
    
    local k,l,m
    for k=0,math.ceil((dsize[1]-border*2)/adj_fov)-1 do
        for l=0,math.ceil((dsize[2]-border*2)/adj_fov)-1 do
            for m=0,math.ceil((dsize[3]-border*2)/adj_fov)-1 do
                collectgarbage()
                local corner={k*adj_fov+1+shift,l*adj_fov+1+shift,m*adj_fov+1+shift}
                
                --TODO: use black padding ?
                corner[1]=math.min(corner[1],dsize[1]-fov+1) 
                corner[2]=math.min(corner[2],dsize[2]-fov+1)
                corner[3]=math.min(corner[3],dsize[3]-fov+1)
                
                get_tile(_minibatch, in_volumes, border, fov, corner, features)
               
                local outputs = model:forward(_minibatch)
                
                put_tile_max( out_volume, outputs, classes, border, fov, corner )

                if out_raw then
                    put_tile_feature( out_raw, outputs, classes, border, fov, corner, 2)
                end
            end
        end
    end
    collectgarbage()
    
    return out_volume
end


--kate: indent-width 4; replace-tabs on; show-tabs on;hl lua
