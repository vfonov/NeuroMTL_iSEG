-------------------------------- MNI Header -----------------------------------
-- @NAME       :  cnn_random_fcn_ace_iseg_s2_final.lua
-- @DESCRIPTION:  Final training for iSEG challenge submission
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
require 'optim'
require 'nn'
require 'xlua'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'paths'

local math=require 'math'
require 'minc2_simple'

-- for progress visualization
display = require 'display'

paths.dofile('kappa.lua')
paths.dofile('model.lua')
paths.dofile('data.lua')
paths.dofile('report.lua')
paths.dofile('tiles.lua')
paths.dofile('dpt_util.lua')
paths.dofile('train.lua')
paths.dofile('validate.lua')

torch.setdefaulttensortype('torch.FloatTensor')

-- setup input data
local work='temp_iseg_priors/'


local training_list="iSEG_sym_train_list.lst"
local training_prefix="data/"

local classes=4

local lm={}
local remap=nil

-- convert training library samples to expected format (i.e strip extras)
local _samples={}
local i,j
local use_original=false -- use generated data


if use_original then
    -- convert to full path
    for i,j in ipairs(library.library) do
        _samples[#_samples+1]={_prefix..j[1], _prefix..j[2]}
    end
else
    --remap=map
    _samples=load_csv(training_prefix..training_list,training_prefix)
end

-- create spatial priors
--all_samples=prepare_data_with_priors_coordinates_simple(_samples, work)
all_samples=_samples
local label_map={['1']='CSF',['2']='GM',['3']='WM'}
local map={1,2,3} -- identity

if label_map==nil then
    local i,j
    if map ~=nil then
        for i,j in pairs(map) do
            lm[i]=string.format('%d',j)
        end
    else
        for i=1,classes-1  do
            lm[i]=string.format('%d',i)
        end
    end
else
    for i,j in pairs(map) do
        lm[i]=label_map[string.format('%d',j)]
    end
end

print(string.format("Processing %d classes",classes))
print(lm)

local output_template='iseg_train/v2_final_%d_fold_'
local pretrain_model_template='iseg_train/v2_s0_1_fold_training.t7'  -- pretrained model
local pretrain_single=true
local discrete_features={false,false}

local model_opts = { -- replicates cnn_random_fcn_cerebellum_HCP_spatial.lua
    -- U-Net layers
    config={
        { map_in=2,   map_out=64, l1=5, l2=5,  ds=2, bypass=true, up=5, input_layers=2,  contiguous=true},
        { map_in=16,  map_out=64, l1=5, l2=5,  ds=2, bypass=true, up=3, contiguous=true},
        { map_in=16,  map_out=64,  l1=5, l2=3, ds=2, bypass=true, up=3, contiguous=true},
        { map_in=16,  map_out=64,  l1=3, l2=3, ds=2, bypass=true, up=3, contiguous=true},
        { map_in=32,  map_out=64,  l1=1, l2=3, l3=5 },
        -- inner module, average information from largest  fov  map_in=32 - seem to be able to reach kappa of 0.9
    },
    
    -- final "flat" layers
    config2= {
      {{m=64,l=3}},
      {{m=32,l=1},dropout=0.5},
      {{m=classes,l=1,relu=false}}
    },

    classes=classes,
    features=2, -- 2 x itensity 

    border=8,
    fov=80,

    nGPU=1,
    folds=1,
    -- 
    start_fold=1,
    run_folds=1,
    discrete_features=discrete_features
}


--
random_opt={
    rot=5, scale=0.02, shift=2, gain=0.001, intensity=0.0001,
    add_random_samples=100,
    resample=2,
    nthread=4,
    order={2,2,2},
    discrete=discrete_features
}

validation=0

local program={
    { 
      batches=4000, iter=2, LR=0.0001 ,LRd=0.0001, beta1=0.9, beta2=0.999, eps=1e-8, samples=3, validate=20, optim=optim.adam, rehash=200, regenerate=400,
      momentum=0.9,  -- momentum
      WD=0.0001,       -- weight decay
      nesterov=false
    }
}

-- seed RNG
torch.manualSeed(4)

--all_samples=_samples
do
    local reference=minc2_file.new(all_samples[1][1])
    model_opts.data_size =   reference:volume_size()
end

print(string.format("Input data size=%s",model_opts.data_size))
print(string.format("Total number of samples=%dx%d",#all_samples,#(all_samples[1])))

local fold

for fold=model_opts.start_fold-1,(model_opts.run_folds-1) do
    collectgarbage()
    local model_variant=string.format(output_template,fold+1)
    local pretrain_model=pretrain_model_template
    
    if not pretrain_single then
        pretrain_model=string.format(pretrain_model_template,fold+1)
    end

    local testing_range_lo,testing_range_hi
    if model_opts.folds>1 then
        testing_range_lo=fold*math.floor(#all_samples/model_opts.folds)+1
        testing_range_hi=(fold+1)*math.floor(#all_samples/model_opts.folds)
    else
        testing_range_lo=0
        testing_range_hi=0
    end
   
    local test_samples  = subrange(all_samples,testing_range_lo,testing_range_hi)
    local train_samples = subrange(all_samples,1,testing_range_lo-1)
    local i
    for i=testing_range_hi+1, #all_samples do -- add leftovers
        train_samples[#train_samples+1]=all_samples[i]
    end
    print(string.format("model_opts.features=%d",model_opts.features))
    
    local mean_sd=calculate_mean_sd_files(train_samples,model_opts.features)

    -- remove validation_samples
    local validation_samples=subrange(train_samples,1,validation)
    train_samples=subrange(train_samples,validation+1,#train_samples)
    print(train_samples)

    -- parameters for random samples
    random_opt.mean_sd=mean_sd

    -- model,criterion=make_model(fov,maps1,maps2,l1,l2,s1,s2,HUs,classes)
    local model,criterion=make_model_unet_mk2( model_opts )
    
    -- additional reset to make sure model is in default state between folds
    for i,module in ipairs(model:listModules()) do
        module:reset()
    end

    print(string.format("Border=%d",model_opts.border))
    print(string.format("Train dataset:%d validation dataset:%d testing dataset:%d", #train_samples, #validation_samples, #test_samples))
    print(model)
    
    final_model_name=model_variant..'training.t7'
    
    if pretrain_model~=nil and paths.filep(pretrain_model) then
            local model_pre=torch.load(pretrain_model)
            print("Loaded pretrained model:"..pretrain_model)
            model_pre=model_pre:cuda()
            print(model_pre)
            model=model_pre
            -- TODO: add support for loading paralel GPU model
    end
    
    if model_opts.nGPU>1 then
        model = makeDataParallel(model, model_opts.nGPU)
    end
    
    
    model=train_fcn( {
                model=model,
                criterion=criterion,
                model_opts=model_opts,
                random_opt=random_opt,
                program=program,
                model_variant=model_variant,
                train_samples=train_samples,
                validation_samples=validation_samples,
                temp_dir=string.format("temp/%02d",fold), -- TODO: use work directory?
                mean_sd=mean_sd,
                fold=fold,
                model_variant=model_variant
                }  )
    
      
    if model_opts.nGPU>1 then
        model=cleanDataParallel(model) -- remove data paralell table
    end
    
    -- TODO: convert to NN and CPU ? 
    model.mean_sd=mean_sd
    model.classes=model_opts.classes
    model.fov=model_opts.fov
    model.border=model_opts.border
    model.features=model_opts.features
    model.discrete_features=model_opts.discrete_features
    model.simple_spatial=true
    
    torch.save(final_model_name, model)
    ----
    
    if #test_samples>0 then
        validate({
            model=model,
            model_opts=model_opts,
            test_samples=test_samples,
            model_variant=model_variant,
            mean_sd=mean_sd,
            fold=fold,
            lm=lm,
            criterion=criterion
        })
    end
    
    -- cleanup memory
    model=nil
    s_model=nil
    trained_model=nil
    collectgarbage()
    collectgarbage()
    collectgarbage()
end
