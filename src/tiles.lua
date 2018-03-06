-------------------------------- MNI Header -----------------------------------
-- @NAME       :  tiles.lua
-- @DESCRIPTION:  overlapping tiles sampler
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
local math=require 'math'

function get_samples_fcn_r_ss(minibatch, dataset, samples, state, border, fov, features)
    local border=border or 0
    local sz=torch.LongTensor(3)
    sz:fill(fov)
    local adj_fov=fov-border*2
    local volume_sz=dataset[1][2]:size()
    local pidx={{1,1},{1,1},{1,1}}
    local pidx2={{1,1},{1,1},{1,1}}
    
    if state.shuffle == nil then
        state.shuffle=torch.randperm(#dataset):long()
        state.idx=1
        
        state.dims=torch.LongTensor(4)
        state.dims[4]=#dataset
        state.dims[3]=math.ceil((volume_sz[3]-border*2)/adj_fov)
        state.dims[2]=math.ceil((volume_sz[2]-border*2)/adj_fov)
        state.dims[1]=math.ceil((volume_sz[1]-border*2)/adj_fov)
        
        state.stride=torch.LongTensor(4)
        state.stride[1]=1
        state.stride[2]=state.dims[1]
        state.stride[3]=state.dims[2]*state.stride[2]
        state.stride[4]=state.dims[3]*state.stride[3]
        
        state.length=state.dims:prod()
        state.shuffle=torch.randperm(state.length):long()
        state.idx=1
    end
    
    local storage_sz=(adj_fov)*(adj_fov)*(adj_fov)
    local s
    local dx={torch.random(-fov/2,fov/2),torch.random(-fov/2,fov/2),torch.random(-fov/2,fov/2)}
    local n_features=1
    for s=1,samples do
        local p=state.shuffle[state.idx]-1 -- damn 1-based arrays
        state.idx=state.idx % state.length + 1 
        local id=math.floor(p/state.stride[4])+1
        
        p=p%state.stride[4]
        local k = math.max(math.min( math.floor(p/state.stride[3])*adj_fov+1+dx[1], volume_sz[3]-fov+1),1)
        p=p%state.stride[3]
        local j = math.max(math.min( math.floor(p/state.stride[2])*adj_fov+1+dx[2], volume_sz[2]-fov+1),1)
        p=p%state.stride[2]
        local i = math.max(math.min( p*adj_fov+1                            +dx[3], volume_sz[1]-fov+1),1)
        
        pidx[1]={i,i+fov-1}
        pidx[2]={j,j+fov-1}
        pidx[3]={k,k+fov-1}
        
        pidx2[1]={i+border,i+fov-1-border}
        pidx2[2]={j+border,j+fov-1-border}
        pidx2[3]={k+border,k+fov-1-border}
        
        minibatch[1][ { s,   1,{},{},{} } ]:copy(dataset[ id ][1][pidx])
        
        if features then
            local f
            for f=1,#features do
                minibatch[1][ { s,   f+1,{},{},{} } ]:copy(features[f][pidx])
            end
        end
        
        local b= dataset[ id ][2] [pidx2]:contiguous() -- labels
        minibatch[2][ { {(s-1)*storage_sz+1,s*storage_sz} } ]:copy(b:view(storage_sz))
        
        state.idx = state.idx % state.length + 1 
    end
end



function get_sample_fcn_ss(minibatch, dataset, border, fov, corner, features)
    
    local corner=corner or {1,1,1}
    local sz=dataset[1]:size()
    local pidx={{corner[1],corner[1]+fov-1},{corner[2],corner[2]+fov-1},{corner[3],corner[3]+fov-1}}
    local adj_fov=fov-border*2
    local b_pidx={{corner[1]+border,corner[1]+fov-border-1},{corner[2]+border,corner[2]+fov-border-1},{corner[3]+border,corner[3]+fov-border-1}}
    local storage_sz=(adj_fov)*(adj_fov)*(adj_fov)
    
    minibatch[1][ { 1, 1,{},{},{} } ]:copy(dataset[1][pidx])
    
    if features then
        local f
        for f=1,#features do
            minibatch[1][ { 1, f+1,{},{},{} } ]:copy(features[f][pidx])
        end
    end
        
    local b=dataset[2][b_pidx]:contiguous()
    minibatch[2][ { {1,storage_sz} } ]:copy(b:view(storage_sz))
end


function allocate_tiles_fcn_ss(samples, border, fov, features)
    local features=features or 1
    local volume_sz=torch.LongTensor(3)
    volume_sz:fill(fov)
    local out_size=volume_sz:clone()
    
    border=border or 0
    out_size:add(-2*border)
    
    local out_sz=out_size[1]*out_size[2]*out_size[3]
    
    local out1=torch.CudaTensor( samples,features,volume_sz[1],volume_sz[2],volume_sz[3] )
    local out2=torch.CudaByteTensor( samples*out_sz )
    
    return {out1,out2},out_size
end

function allocate_tiles_fcn_ss_auto(samples, border, fov, features)
    local features=features or 1
    local volume_sz=torch.LongTensor(3)
    volume_sz:fill(fov)
    local out_size=volume_sz:clone()
    
    border=border or 0
    out_size:add(-2*border)
    
    local out_sz=out_size[1]*out_size[2]*out_size[3]
    
    local out1=torch.CudaTensor( samples,features,volume_sz[1],volume_sz[2],volume_sz[3] )
    local out2=torch.CudaTensor( samples*out_sz )
    
    return {out1,out2},out_size
end


function put_tiles_max_fcn_ss(ds, out, classes,border,fov,corner)
    local _,m
    
    local adj_fov  = fov-border-1
    local adj_fov2 = fov-border*2
    -- ds[{}]=1
    local _,m=out:float():view(adj_fov2, adj_fov2, adj_fov2, classes):max(4)
    
    ds[ {{corner[1]+border,corner[1]+adj_fov},
         {corner[2]+border,corner[2]+adj_fov},
         {corner[3]+border,corner[3]+adj_fov}} ] = m:add(-1)

end


function put_tiles_fcn_ss(ds, out, classes,c, border, fov, corner)
    local adj_fov=fov-border-1
    local adj_fov2=fov-border*2
    
    ds[ {{corner[1]+border,corner[1]+adj_fov},
         {corner[2]+border,corner[2]+adj_fov},
         {corner[3]+border,corner[3]+adj_fov}} ] = 
        out:float():view(adj_fov2,adj_fov2,adj_fov2, classes)[{{},{},{},c}]
end


function get_samples_fcn_random(minibatch, dataset, samples, state, border, fov, features)
    local border=border or 0
    local sz=torch.LongTensor(3)
    sz:fill(fov)
    local adj_fov=fov-border*2
    
    local volume_sz=dataset[1][2]:size()
    local pidx={{1,1},{1,1},{1,1}}
    local pidx2={{1,1},{1,1},{1,1}}
    local features=features or 1
    
    if state.shuffle == nil then
        state.shuffle=torch.randperm(#dataset):long()
        state.idx=1
        print("reshuffled")
        print()
    end
    
    local storage_sz=(adj_fov)*(adj_fov)*(adj_fov)
    local s
    --local dx={torch.random(-fov/2,fov/2),torch.random(-fov/2,fov/2),torch.random(-fov/2,fov/2)}
    
    for s=1,samples do
        local id=state.shuffle[state.idx] 
        state.idx=state.idx % (#dataset) + 1 
        
        local i=torch.random(1,volume_sz[1]-fov+1)
        local j=torch.random(1,volume_sz[2]-fov+1)
        local k=torch.random(1,volume_sz[3]-fov+1)
        
        pidx[1]={i,i+fov-1}
        pidx[2]={j,j+fov-1}
        pidx[3]={k,k+fov-1}

        pidx2[1]={i+border,i+fov-1-border}
        pidx2[2]={j+border,j+fov-1-border}
        pidx2[3]={k+border,k+fov-1-border}
        
        local f
        for f=1,features do
          --print(f,pidx)
          minibatch[1][ { s, f,{},{},{} } ]:copy(dataset[ id ][f][pidx])
        end
        
        local b= dataset[ id ][features+1] [pidx2]:contiguous()
        minibatch[2][ { {(s-1)*storage_sz+1,s*storage_sz} } ]:copy(b:view(storage_sz)):add(1) 
    end
end

function get_sample_fcn_files(minibatch, sample_files, border, fov, corner, features, mean_sd, discrete_features)
    
    local corner=corner or {1,1,1}
    local pidx={{corner[1],corner[1]+fov-1},{corner[2],corner[2]+fov-1},{corner[3],corner[3]+fov-1}}
    local adj_fov=fov-border*2
    local b_pidx={{corner[1]+border,corner[1]+fov-border-1},{corner[2]+border,corner[2]+fov-border-1},{corner[3]+border,corner[3]+fov-border-1}}
    local storage_sz=(adj_fov)*(adj_fov)*(adj_fov)
    
    
    local f
    for f=1,features do
        local _minc=minc2_file.new( sample_files[f] )
        _minc:setup_standard_order()
        local _vol
        
        if discrete_features and not discrete_features[f] then
            _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT,pidx)
            _vol=(_vol- mean_sd.mean[f])/mean_sd.sd[f] 
        else
            _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT,pidx) -- TODO: use integer here?
        end
        minibatch[1][ { 1, f ,{},{},{} } ]:copy(_vol)
    end
        
    local _minc2=minc2_file.new(sample_files[features+1])
    _minc2:setup_standard_order()
    
    local b=_minc2:load_hyperslab(minc2_file.MINC2_INT,b_pidx)
    b:add(1)
    --b:mul(-1):add(3)
    minibatch[2][ { {1,storage_sz} } ]:copy(b:view(storage_sz))
end


function get_samples_fcn_random_files(minibatch, sample_files, samples, state, border, fov, features, mean_sd, discrete_features)
    local border=border or 0
    local sz=torch.LongTensor(3)
    
    sz:fill(fov)
    local adj_fov=fov-border*2
    
    --local volume_sz=sample_files[1][2]:size()
    local features=features or 1
    
    if state.shuffle == nil then
        state.shuffle=torch.randperm(#sample_files):long()
        state.idx=1
    end
    
    local storage_sz=(adj_fov)*(adj_fov)*(adj_fov)
    local s
    
    for s=1,samples do
        local id=state.shuffle[state.idx] 
        state.idx=state.idx % (#sample_files) + 1 
        
        local pidx={{1,1},{1,1},{1,1}}
        local pidx2={{1,1},{1,1},{1,1}}
        
        for f=1,features do
            local _minc=minc2_file.new(sample_files[id][f])
            _minc:setup_standard_order()

            if f==1 then
                volume_sz=_minc:volume_size()
                
                local i=torch.random(1,volume_sz[1]-fov+1)
                local j=torch.random(1,volume_sz[2]-fov+1)
                local k=torch.random(1,volume_sz[3]-fov+1)
                
                pidx[1]={i,i+fov-1}
                pidx[2]={j,j+fov-1}
                pidx[3]={k,k+fov-1}

                pidx2[1]={i+border,i+fov-1-border}
                pidx2[2]={j+border,j+fov-1-border}
                pidx2[3]={k+border,k+fov-1-border}
            end
            local _vol
            if discrete_features and not discrete_features[f] then
               _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT,pidx)
               _vol=(_vol-mean_sd.mean[f])/mean_sd.sd[f]
            else
               _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT,pidx) -- TODO: use int?
            end
            minibatch[1][ { s, f,{},{},{} } ]:copy(_vol)
        end
        
        local _minc2=minc2_file.new(sample_files[id][features+1])
        _minc2:setup_standard_order()
        
        local b= _minc2:load_hyperslab(minc2_file.MINC2_INT,pidx2) --TODO: use short?
        b:add(1)
        --b:mul(-1):add(3)
        
        minibatch[2][ { {(s-1)*storage_sz+1,s*storage_sz} } ]:copy(b:view(storage_sz))
    end
end


function get_samples_fcn_random_files_auto(minibatch, sample_files, 
        samples, state, border, fov, features, 
        mean_sd, discrete_features, random_opt)
    --
    local border=border or 0
    local sz=torch.LongTensor(3)
    local corrupt=random_opt.corrupt or 0.0
    local intensity_noise=random_opt.intensity or 0.0
    
    sz:fill(fov)
    local adj_fov=fov-border*2
    
    --local volume_sz=sample_files[1][2]:size()
    local features=features or 1
    
    if state.shuffle == nil then
        state.shuffle=torch.randperm(#sample_files):long()
        state.idx=1
    end
    
    local storage_sz=(adj_fov)*(adj_fov)*(adj_fov)
    local s
    
    for s=1,samples do
        local id=state.shuffle[state.idx] 
        state.idx=state.idx % (#sample_files) + 1 
        
        local pidx={{1,1},{1,1},{1,1}}
        local pidx2={{1,1},{1,1},{1,1}}
        local _corrupt
        
        if corrupt>0.0 then
          local _c_sum=0
          while _c_sum<2 do
             _corrupt=torch.lt(torch.rand(fov,fov,fov),corrupt)
             _c_sum=fov*fov*fov-torch.sum(_corrupt)
          end
        end
        
        for f=1,features do
            local _minc=minc2_file.new(sample_files[id][f])
            _minc:setup_standard_order()

            if f==1 then
                volume_sz=_minc:volume_size()
                
                local i=torch.random(1,volume_sz[1]-fov+1)
                local j=torch.random(1,volume_sz[2]-fov+1)
                local k=torch.random(1,volume_sz[3]-fov+1)
                
                pidx[1]={i,i+fov-1}
                pidx[2]={j,j+fov-1}
                pidx[3]={k,k+fov-1}

                pidx2[1]={i+border,i+fov-1-border}
                pidx2[2]={j+border,j+fov-1-border}
                pidx2[3]={k+border,k+fov-1-border}
            end
            
            local _vol
            if discrete_features and not discrete_features[f] then
               _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT,pidx)
               _vol=(_vol-mean_sd.mean[f])/mean_sd.sd[f]
               
               -- add gaussian noise
               if intensity_noise>0.0 and f==1 then
                   _vol=_vol+torch.randn(fov,fov,fov):float()*intensity_noise
               end
               
               -- zero-out corrupted voxels!
               if corrupt>0.0 and f==1 then
                 _vol[_corrupt]=0
               end
               
            else
               _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT,pidx) -- TODO: use int?
            end
            
            minibatch[1][ { s, f,{},{},{} } ]:copy(_vol)
        end
        
        local _minc2=minc2_file.new(sample_files[id][1])
        _minc2:setup_standard_order()
        
        local b= _minc2:load_hyperslab(minc2_file.MINC2_FLOAT,pidx2) --TODO: use short?
        -- apply same normalization
        b=(b-mean_sd.mean[1])/mean_sd.sd[1]
        minibatch[2][ { {(s-1)*storage_sz+1,s*storage_sz} } ]:copy(b:view(storage_sz))
    end
end


function get_volumes_files(sample_files, features, mean_sd, discrete_features)
    local volumes={}
    
    local f
    for f=1,features do
        local _minc=minc2_file.new( sample_files[f] )
        _minc:setup_standard_order()
        local _vol
        
        if discrete_features and not discrete_features[f] then
            _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT)
            _vol=(_vol- mean_sd.mean[f])/mean_sd.sd[f] -- TODO: add per feature mean/sd
        else
            _vol=_minc:load_hyperslab(minc2_file.MINC2_FLOAT) -- TODO: use integer here?
        end
        volumes[#volumes+1]=_vol
    end
    
    if sample_files[features+1]~= nil then
        local _minc2=minc2_file.new(sample_files[features+1])
        _minc2:setup_standard_order()
        
        local b=_minc2:load_hyperslab(minc2_file.MINC2_INT)
        b:add(1)
        volumes[#volumes+1]=b
    else
        volumes[#volumes+1]=nil
    end

    return volumes
end

--kate: indent-width 4; replace-tabs on; show-tabs on;hl lua
