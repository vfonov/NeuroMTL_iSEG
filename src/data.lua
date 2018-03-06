-------------------------------- MNI Header -----------------------------------
-- @NAME       :  data.lua
-- @DESCRIPTION:  loading and saving data 
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
require 'io'
require 'torch'
--require 'cutorch'
require 'paths'
require 'sys'
require 'minc2_simple'
require 'xlua'
local math=require 'math'


function my_execute(command)
    local r,n,c=os.execute(command)
    assert(c==0,"Error running:"..command.."\nCode:"..c)
end

function load_csv(csv_file,prefix)
    -- load training list
    local samples={}
    for line in io.lines(csv_file) do
        local sample={}
        for i,j in pairs(string.split(line,",")) do
            sample[#sample+1]=prefix..j
        end
        samples[#samples + 1] = sample
    end
    return samples
end

function load_csv_labels(csv_file,prefix)
    -- load training list, one input image , several factors, last column - predicted
    local samples={}
    local line
    for line in io.lines(csv_file) do
        local sample={}
        for i,j in pairs(string.split(line,",")) do
          if i<2 then
            sample[#sample+1]=prefix..j
          else
            sample[#sample+1]=tonumber(j)
          end
        end
        samples[#samples + 1] = sample
    end
    return samples
end


-- load minc files into memory
function load_data(samples,prefix,show_progress,display_on) 
    local dataset={}
    local i,l
    
    local display_samples = {
        title = "Samples",
        labels = {"Sample", "mean","sd"},
        ylabel = "V"
    }

    local sample_data={}


    for i,l in pairs(samples) do
        
        local f1=l[1]
        local s1=l[2]
        
        if prefix~=nil then
            f1=paths.concat(prefix,f1)
            s1=paths.concat(prefix,s1)
            nl_xfm=paths.concat(prefix,s1)
        end

        local t1=minc2_file.new(f1)
        t1:setup_standard_order()
        
        local seg=minc2_file.new(s1)
        seg:setup_standard_order()
        
        local vol_t1=t1:load_complete_volume(minc2_file.MINC2_FLOAT)
        local vol_seg=seg:load_complete_volume(minc2_file.MINC2_INT)
        
        dataset[#dataset+1]={ vol_t1, 
                              vol_seg } -- converted to 1-based info
        t1:close()   -- make sure files are closed now
        seg:close()
        if show_progress then
            xlua.progress(i,#samples)
        end
        if display_on then
             table.insert(sample_data,{i,torch.mean(vol_t1),torch.std(vol_t1)})
             display_samples.win = display.plot(sample_data, display_samples)
        end
    end
    return dataset
end

function load_data_labels(samples,prefix) 
    local dataset={}
    local _,l
    for _,l in pairs(samples) do
        local f1=l[1]
        
        if prefix then
            f1=paths.concat(prefix,f1)
        end

        local t1=minc2_file.new(f1)
        t1:setup_standard_order()
        local img=t1:load_complete_volume(minc2_file.MINC2_FLOAT)
        local sample={ img }
        t1:close() --make sure files are closed now
        
        for j,k in ipairs(l) do
          if j>1 then
            local f=img:clone()
            f:fill(k)
            sample[#sample+1]=f
          end
        end
        dataset[#dataset+1]=sample
    end
    return dataset
end

function load_data_from_grad_library(samples,prefix) 
    local dataset={}
    local _,l
    for _,l in pairs(samples) do
        -- print(string.format("Opening %s %s",l[1],l[2]))
        local f1=l[3]
        
        if prefix then
            f1=paths.concat(prefix,f1)
        end

        local t1=minc2_file.new(f1)
        t1:setup_standard_order()
        local img=t1:load_complete_volume(minc2_file.MINC2_FLOAT)
        local sample={ img }
        t1:close() --make sure files are closed now
        
        local f=img:clone()
        f:fill(tonumber(l[1])+1)
        sample[#sample+1]=f
        
        dataset[#dataset+1]=sample
    end
    return dataset
end


function load_data1(samples,prefix) 
    local dataset={}
    local _,l
    for _,l in pairs(samples) do
--        print(string.format("Opening %s %s",l[1],l[2]))
        
        local f1=l[1]
        
        if prefix then
            f1=paths.concat(prefix,f1)
        end

        local t1=minc2_file.new(f1)
        t1:setup_standard_order()
        
        
        dataset[#dataset+1]=t1:load_complete_volume(minc2_file.MINC2_FLOAT)
        t1:close()
    end
    return dataset
end


function calculate_mean_sd(dataset)
    local _mean=0.0
    local _sd=0.0
    local j
    
    for j=1,(#dataset) do
        _mean=_mean+torch.mean(dataset[j][1])
        _sd=_sd+torch.std(dataset[j][1])
    end

    _mean=_mean/#dataset
    _sd=_sd/#dataset
   
    return {mean=_mean,sd=_sd}
end

function calculate_mean_sd_files(samples, features,ss)
    local _mean=torch.DoubleTensor(features)
    local _sd=torch.DoubleTensor(features)
    
    local j,f
    local features = features or 1
    _sd:fill(0)
    _mean:fill(0)
    
    ss=ss or #samples
    -- TODO: add shuffle if ss<#samples
    for j=1,ss do
        for f=1,features do
            local _minc=minc2_file.new(samples[j][f])
            _minc:setup_standard_order()
            local _vol=_minc:load_complete_volume(minc2_file.MINC2_FLOAT)
            
            _mean[f]=_mean[f]+torch.mean(_vol)
            _sd[f]=_sd[f]+torch.std(_vol)
        end
    end

    for f=1,features do
        _mean[f]=_mean[f]/ss
        _sd[f]=_sd[f]/ss
    end
    
    print(_sd)
    
    return {mean=_mean:float(),sd=_sd:float()}
end

function normalize(dataset,mean_sd)
    if dataset==nil then
        return nil
    end
    local _mean=mean_sd.mean or 0.0
    local _sd=mean_sd.sd or 1.0
    local j
    local new_dataset={}
    
    for j=1,(#dataset) do
        new_dataset[j]={}
        local i,k
        for i,k in ipairs(dataset[j]) do
          new_dataset[j][i]=dataset[j][i]:clone()
        end
        
        new_dataset[j][1]=(new_dataset[j][1]-_mean)/_sd
    end
    return new_dataset
end

function normalize0(dataset,mean_sd)
    local j
    local new_dataset={}
    
    for j=1,(#dataset) do
        new_dataset[j]={}
        local i,k
        for i,k in ipairs(dataset[j]) do
          new_dataset[j][i]=dataset[j][i]:clone()
        end
        local _mean=torch.mean(new_dataset[j][1])
        local _sd=torch.std(new_dataset[j][1])
        new_dataset[j][1]=(new_dataset[j][1]-_mean)/_sd
    end
    return new_dataset
end

function normalize1(dataset,mean_sd)
    local _mean=mean_sd.mean or 0.0
    local _sd=mean_sd.sd or 1.0
    local j
    local new_dataset={}
    
    for j=1,(#dataset) do
        new_dataset[j]={}
        new_dataset[j]=dataset[j]:clone()
        new_dataset[j]=(new_dataset[j]-_mean)/_sd
    end
    return new_dataset
end


-- helper function to extract slice from table
function subrange(t, first, last)
  local sub = {}
  local i
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

function load_json_library(l)
   local cjson = require "cjson"
   local f = io.open(l, "rb")
   assert(f~=nil,"Error opening file:"..l)
   local content = f:read("*all")
   f:close()
   
   r=cjson.decode(content)
   if r.label_map == cjson.null then r.label_map=nil end
   if r.map == cjson.null then r.map=nil end
   return r
end


function prepare_data_with_priors(library,data_prefix,work,nthread)
    local nthread=nthread or 1
    local samples=library.library
    
    my_execute('mkdir -p ' .. work)
    local threads = require 'threads'
    local pool = threads.Threads(nthread)
    local out_samples={}
    local _order=0
    local _baa='' -- '--baa'
    local atlas=data_prefix..library.local_model_seg
    

    for i,j in ipairs(samples) do
        --
        local nl_xfm=data_prefix..j[3]
        local prior=work .. 'prior_'..j[1]
        local _sample={ data_prefix..j[1], prior, data_prefix..j[2] }
        local my_execute=my_execute -- make local?
        out_samples[#out_samples+1]=_sample
        if not paths.filep(prior) then
            pool:addjob(
                function()
                    local cmd=string.format('itk_resample --invert_transform --byte --labels --order %d %s %s --transform %s --clob %s',_order, atlas, prior, nl_xfm,_baa)
                    --print(cmd)
                    my_execute(cmd)
                end )
        end
    end
    pool:synchronize()
    return out_samples
end


function prepare_data_with_priors_one_hot(library, data_prefix, nclasses, work, nthread)
    local nthread=nthread or 1
    local samples=library.library
    
    my_execute('mkdir -p ' .. work)
    local threads = require 'threads'
    local pool = threads.Threads(nthread)
    local out_samples={}
    local _order=0
    local _baa='' -- '--baa'
    local atlas=data_prefix..library.local_model_seg
    

    for i,j in ipairs(samples) do
        --
        local dname=paths.basename(j[1],'.mnc')
        local nl_xfm=data_prefix..j[3]
        
        local prior=work .. 'prior_'..dname..'.mnc'
        local priors={}  
        
        local _sample={ data_prefix..j[1]}  
        local k
        
        local my_execute=my_execute -- make local?
        
        for k=1,nclasses do
            _sample[#_sample+1]=string.format('%s%s_%02d.mnc',work,dname,k-1)
        end
        
        _sample[#_sample+1]=data_prefix..j[2]  -- ground truth
        
        out_samples[#out_samples+1]=_sample
        if not paths.filep(prior) then
            pool:addjob(
                function()
                    my_execute(string.format('itk_resample --invert_transform --byte --labels --order %d %s %s --transform %s --clob %s',_order, atlas, prior, nl_xfm, _baa))
                    my_execute(string.format('itk_split_labels --byte --antialias --expit 1.0 --clob --normalize %s %s',prior,work..dname..'_%02d.mnc'))
                end )
        end
    end
    pool:synchronize()
    return out_samples
end


function dump_volume(fname,ref,out_tensor)
        local out_minc=minc2_file.new()
        if ref then
          out_minc:define(ref:store_dims(), minc2_file.MINC2_BYTE,  minc2_file.MINC2_FLOAT)
        else
          local volume_sz=out_tensor:size()
          
          local my_dims={
               {id=minc2_file.MINC2_DIM_X,  length=volume_sz[3],start=0.0,  step=1.0},
               {id=minc2_file.MINC2_DIM_Y,  length=volume_sz[2],start=0.0,  step=1.0},
               {id=minc2_file.MINC2_DIM_Z,  length=volume_sz[1],start=0.0, step=1.0}
          }          
          out_minc:define(my_dims, minc2_file.MINC2_BYTE,  minc2_file.MINC2_FLOAT)
        end
        out_minc:create(fname)
        out_minc:setup_standard_order()
        out_minc:save_complete_volume(out_tensor)
        out_minc:close()
end



function prepare_data_with_priors_coordinates(library, data_prefix, nclasses, work, nthread)
    local nthread=nthread or 1
    local samples=library.library
    
    my_execute('mkdir -p ' .. work)
    local threads = require 'threads'
    local pool = threads.Threads(nthread)
    local out_samples={}
    local _order=0
    local _baa='' -- '--baa'
    local atlas=data_prefix..library.local_model_seg
    
    local coords={work .. 'x.mnc',work .. 'y.mnc',work .. 'z.mnc'}
    
    local ref=minc2_file.new(atlas)
    ref:setup_standard_order()
    
    local vol_ref=ref:load_complete_volume(minc2_file.MINC2_FLOAT)
    
    local sp1=torch.linspace(-1,1,vol_ref:size(1))
    local sp2=torch.linspace(-1,1,vol_ref:size(2))
    local sp3=torch.linspace(-1,1,vol_ref:size(3))
    
    local i,j
    
    for i=1,vol_ref:size(2) do
        for j=1,vol_ref:size(3) do
            vol_ref[{{},{i},{j}}]=sp1
        end
    end
    dump_volume(coords[1],ref,vol_ref)
    for i=1,vol_ref:size(1) do
        for j=1,vol_ref:size(3) do
            vol_ref[{{i},{},{j}}]=sp2
        end
    end
    dump_volume(coords[2],ref,vol_ref)
    for i=1,vol_ref:size(1) do
        for j=1,vol_ref:size(2) do
            vol_ref[{{i},{j},{}}]=sp3
        end
    end
    dump_volume(coords[3],ref,vol_ref)
    
    
    for i,j in ipairs(samples) do
        --
        local dname=paths.basename(j[1],'.mnc')
        local nl_xfm=data_prefix..j[3]
        
        local prior=work .. 'prior_c_'..dname..'.mnc'
        local priors={}  
        
        local _sample={ data_prefix..j[1]}  
        local k
        
        local my_execute=my_execute -- make local?
        
        for k=1,3 do
            _sample[#_sample+1]=string.format('%s%s_%d_c.mnc',work,dname,k-1)
        end
        
        if #j>1 then
         _sample[#_sample+1]=data_prefix..j[2]  -- ground truth
        end
        
        out_samples[#out_samples+1]=_sample
        if not paths.filep(_sample[4]) then
            pool:addjob(
                function()
                    for k=1,3 do
                        my_execute(string.format('itk_resample --invert_transform --byte --order %d %s %s --transform %s --clob',_order, coords[k], _sample[k+1], nl_xfm))
                    end
                end )
        end
    end
    pool:synchronize()
    return out_samples
end

function prepare_data_with_priors_coordinates_simple(samples, work)
    --local samples=load_csv(lst,data_prefix)
    
    --my_execute('mkdir -p ' .. work)
    if not paths.dirp(work) then
        paths.mkdir(work)
    end
    
    local out_samples={}
    
    local coords={work .. 'x.mnc',work .. 'y.mnc',work .. 'z.mnc'}
    
    local ref=minc2_file.new(samples[1][1])
    ref:setup_standard_order()
    
    local vol_ref=ref:load_complete_volume(minc2_file.MINC2_FLOAT)
    
    local sp1=torch.linspace(-1,1,vol_ref:size(1))
    local sp2=torch.linspace(-1,1,vol_ref:size(2))
    local sp3=torch.linspace(-1,1,vol_ref:size(3))
    
    local i,j
    
    for i=1,vol_ref:size(2) do
        for j=1,vol_ref:size(3) do
            vol_ref[{{},{i},{j}}]=sp1
        end
    end
    dump_volume(coords[1],ref,vol_ref)
    for i=1,vol_ref:size(1) do
        for j=1,vol_ref:size(3) do
            vol_ref[{{i},{},{j}}]=sp2
        end
    end
    dump_volume(coords[2],ref,vol_ref)
    for i=1,vol_ref:size(1) do
        for j=1,vol_ref:size(2) do
            vol_ref[{{i},{j},{}}]=sp3
        end
    end
    dump_volume(coords[3],ref,vol_ref)
    
    local out_samples={}    
    for i,j in ipairs(samples) do
        local ext_sample={j[1]}
        
        for k=1,3 do
            ext_sample[#ext_sample+1]=coords[k]
        end
        if #j>1 then
         ext_sample[#ext_sample+1]=j[2] -- ground truth, if present
        end
        out_samples[#out_samples+1]=ext_sample
    end
    return out_samples
end

function prepare_data_with_priors_coordinates_simple_stx(samples, work,features)
    --local samples=load_csv(lst,data_prefix)
    local features=features or 1
    --my_execute('mkdir -p ' .. work)
    if not paths.dirp(work) then
        paths.mkdir(work)
    end
    
    local out_samples={}
    
    local coords={work .. 'x.mnc',work .. 'y.mnc',work .. 'z.mnc'}
    
    local ref=minc2_file.new(samples[1][1])
    ref:setup_standard_order()
    local ref_dims=ref:representation_dims()
    
    local vol_ref=ref:load_complete_volume(minc2_file.MINC2_FLOAT)
    -- assume that direction cosines have identity matrix for now
    -- TODO: make it work with arbitrary direction cosines
    -- WARNING: dimensions are 0-based 
    local sp1=torch.linspace(ref_dims[0].start, ref_dims[0].step*ref_dims[0].length+ref_dims[0].start, ref_dims[0].length)
    local sp2=torch.linspace(ref_dims[1].start, ref_dims[1].step*ref_dims[1].length+ref_dims[1].start, ref_dims[1].length)
    local sp3=torch.linspace(ref_dims[2].start, ref_dims[2].step*ref_dims[2].length+ref_dims[2].start, ref_dims[2].length)
    
    local i,j
    
    for i=1,ref_dims[1].length do
        for j=1,ref_dims[2].length do
            vol_ref[{{},{i},{j}}]=sp1
        end
    end
    dump_volume(coords[1],ref,vol_ref)
    for i=1,ref_dims[0].length do
        for j=1,ref_dims[2].length do
            vol_ref[{{i},{},{j}}]=sp2
        end
    end
    dump_volume(coords[2],ref,vol_ref)
    for i=1,ref_dims[0].length do
        for j=1,ref_dims[1].length do
            vol_ref[{{i},{j},{}}]=sp3
        end
    end
    dump_volume(coords[3],ref,vol_ref)
    
    local out_samples={}    
    for i,j in ipairs(samples) do
        local ext_sample={}
        
        for k=1,features do
            ext_sample[#ext_sample+1]=j[k]
        end
        
        for k=1,3 do
            ext_sample[#ext_sample+1]=coords[k]
        end
        
        ext_sample[#ext_sample+1]=j[features+1] -- ground truth, if present
        
        out_samples[#out_samples+1]=ext_sample
    end
    return out_samples
end



function prepare_data_simple(library,data_prefix,work,nthread)
    local out_samples={}

    for i,j in ipairs(samples) do
        --
        local _sample={ data_prefix..j[1], data_prefix..j[2] }
        out_samples[#out_samples+1]=_sample
    end
    return out_samples
end


function generate_random_samples(train_samples, opt, tempdir, model_opts)
    -- apply random translation, scaling, rotation
    local add_random_samples=opt.add_random_samples or 0
    local nthread=opt.nthread
    local features=model_opts.features or model_opts -- TODO: fixe this for backward compatibility?
    --local features=opt.features
    
    if add_random_samples==0 then
        return {}
    end
    
    local nthread=nthread or 1
    local tempdir=tempdir or 'temp' 
    
    my_execute('mkdir -p ' .. tempdir)
    
    local threads = require 'threads'
    local pool = threads.Threads(nthread)
    local i
    local out={}
    
    print(string.format("Features=%d",features))
    print(model_opts.mean_sd.sd)
    
    for i=1,add_random_samples do
        local my_execute=my_execute -- make local?
        
        local s=train_samples[(i-1)%#train_samples+1]
        
        local xfm=string.format('%s/%03d.xfm',tempdir,i)
        local o={}
        --local discrete=opt.dis
        
        for j=1,features do
            if not opt.randomize or opt.randomize[j] then
              o[j]=string.format('%s/%03d.%d.mnc',tempdir,i,j)
            else -- reuse input
              o[j]=s[j]
            end
        end
        
        if opt.classes then
          o[features+1]=string.format('%s/%03d.seg.mnc',tempdir,i)
        end
        
        pool:addjob(
            function()
                my_execute(string.format('param2xfm -clob -translation %f %f %f -rotations %f %f %f -scales %f %f %f %s',
                                        torch.uniform(-opt.shift,opt.shift),torch.uniform(-opt.shift,opt.shift),torch.uniform(-opt.shift,opt.shift),
                                        torch.uniform(-opt.rot,opt.rot),torch.uniform(-opt.rot,opt.rot),torch.uniform(-opt.rot,opt.rot),
                                        1.0+torch.uniform(-opt.scale,opt.scale),1.0+torch.uniform(-opt.scale,opt.scale),1.0+torch.uniform(-opt.scale,opt.scale),
                                        xfm))

                -- apply transformation
                local j

                local baa=""
                if opt.order[features+1]>0 then baa="--baa" end
                
                if opt.classes then
                  local infile_seg =s[features+1]
                  local outfile_seg=o[features+1]
                  my_execute(string.format('itk_resample --byte --labels --order %d  %s %s --transform %s --clob %s', opt.order[features+1], infile_seg, outfile_seg, xfm, baa))
                end

                for j=1,features do
                    local _infile  = s[j]
                    local _outfile = o[j]
                    
                    if not opt.discrete[j] then
                        if not opt.randomize or opt.randomize[j] then
                         if opt.gain==nil or opt.gain==0.0 or model_opts.mean_sd==nil then
                            my_execute(string.format('itk_resample --order %d %s %s --transform %s  --clob',opt.order[j], _infile, _outfile, xfm))
                         else
                            local tmp_res=paths.tmpname()..'.mnc'
                            local tmp_random=paths.tmpname()..'.mnc'
                            
                            local rval=torch.uniform(-opt.intensity, opt.intensity)*model_opts.mean_sd.sd[j]
                            
                            my_execute(string.format('itk_resample --order %d %s %s --transform %s  --clob', opt.order[j], _infile, tmp_res, xfm ))
                            my_execute(string.format('random_volume --float --gauss %f %s %s --clob',opt.gain*model_opts.mean_sd.sd[j], tmp_res, tmp_random ))
                            my_execute(string.format("minccalc -q -clob -express 'A[0]*(1+%f)+A[1]' %s %s %s",rval, tmp_res, tmp_random, _outfile ))
                            
                            os.remove(tmp_res)
                            os.remove(tmp_random)
                         end
                        else
                         -- no need to resample this feature
                        end
                    else
                        my_execute(string.format('itk_resample --labels --order %d %s %s --transform %s --clob %s',opt.order[j], _infile, _outfile, xfm,baa))
                    end
                end
                
            end
        )
        
        out[#out+1]=o
    end

    pool:synchronize()
    --print(out)
    return out
end


function create_spatial_features(train_dataset)
    -- create spatial features
    local features={
        train_dataset[1][1]:clone(),
        train_dataset[1][1]:clone(),
        train_dataset[1][1]:clone()
        }
    local i,j
    
    local sp1=torch.linspace(-1,1,features[1]:size(1))
    local sp2=torch.linspace(-1,1,features[1]:size(2))
    local sp3=torch.linspace(-1,1,features[1]:size(3))
    
    for i=1,features[1]:size(2) do
        for j=1,features[1]:size(3) do
            features[1][{{},{i},{j}}]=sp1
        end
    end
    for i=1,features[1]:size(1) do
        for j=1,features[1]:size(3) do
            features[2][{{i},{},{j}}]=sp2
        end
    end
    for i=1,features[1]:size(1) do
        for j=1,features[1]:size(2) do
            features[3][{{i},{j},{}}]=sp3
        end
    end
    return features
end

--kate: indent-width 4; replace-tabs on; show-tabs on;hl lua
