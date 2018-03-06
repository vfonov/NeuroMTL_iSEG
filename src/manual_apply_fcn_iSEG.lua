-------------------------------- MNI Header -----------------------------------
-- @NAME       :  manual_apply_fcn_iSEG.lua
-- @DESCRIPTION:  apply pre-trained depp-net to t1w/t2w pair
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
require 'paths'
-- require 'xlua'
require 'minc2_simple'

paths.dofile('data.lua')
paths.dofile('apply_fcn.lua')
paths.dofile('model.lua')


function parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch FCN application script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-model',      '',        'Path to model')
   cmd:option('-t1',         '',        'Path to T1 input file')
   cmd:option('-t2',         '',        'Path to T2 input file')
   cmd:option('-out',        '',        'Path to output file')
   cmd:option('-raw',        '',        'Path to output raw file')
   cmd:option('-gpu',        false,     'Use GPU/cudnn')
   cmd:option('-shift',      0,         'Debugging')
   cmd:option('-border',     -1,        'Override border')
   
   cmd:text()
   
   local opt = cmd:parse(arg or {})
   return opt
end

local opt=parse(arg)

if not opt.t1 or not opt.t2 or not opt.out or not opt.model or opt.inp=="" or opt.out=="" or opt.model=="" then
    print(string.format("Need to specify all parameters"))
    return 1
end

if opt.gpu then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
end

local model=torch.load(opt.model)

if opt.gpu then
--    cudnn.convert(model, cudnn)
    --model=model:cuda()
end

--local in_file  =opt.inp
local out_file =opt.out
local out_file_raw=opt.raw

local fov      = model.fov or 96
local border   = model.border or 8
local classes  = model.classes or 2
local mean_sd  = model.mean_sd
local features = 2 -- model.features
local simple_spatial = false -- model.simple_spatial
local discrete_features = {false,false} --model.discrete_features

if features == nil and simple_spatial then features=2 end

if discrete_features == nil then
    discrete_features={}
    for i=1,features do
        discrete_features[i]=false
    end
end

if opt.border ~= -1 then
    border=opt.border
end

model:evaluate()

--local workdir=os.tmpname()
--os.remove(workdir)
--workdir=workdir.."/"

--input_samples=prepare_data_with_priors_coordinates_simple( {{in_file,nil}}, workdir)
input_samples={{opt.t1,opt.t2}}
print(string.format('Features:%d',features))
local in_volumes=get_volumes_files(input_samples[1], features, mean_sd, discrete_features)

local dsize=in_volumes[1]:size()
print(dsize)
local t_out1=in_volumes[1]:clone():zero()
local t_out2=in_volumes[1]:byte():clone():fill(1)

if border ~= model.border then
    --adapt_model_unet_mk2(model,{border=border})
end

if opt.gpu then
    torch.setdefaulttensortype('torch.FloatTensor')
    minibatch=allocate_tile_minibatch_gpu(1, border, fov, features)
else
    torch.setdefaulttensortype('torch.FloatTensor')
    minibatch=allocate_tile_minibatch_cpu(1, border, fov, features)
end

local input_minc=minc2_file.new(opt.t1)

t_out2=apply_fcn{ 
    model=model,
    fov=fov,
    border=border,
    classes=classes,
    features=features, -- TODO: use real features
    in_volumes=in_volumes,
    out_volume=t_out2,
    out_raw=t_out1,
    shift=opt.shift,
    minibatch=minibatch
    }
print("Applied!")

t_out2:add(-1)
--paths.rmall(workdir,"yes")
--print(workdir)

local out_minc=minc2_file.new()
out_minc:define(input_minc:store_dims(), minc2_file.MINC2_BYTE, minc2_file.MINC2_BYTE)
out_minc:create(out_file)
out_minc:setup_standard_order()
out_minc:save_complete_volume(t_out2)
out_minc:close()

if out_file_raw and out_file_raw ~= '' then
    out_minc=minc2_file.new()
    out_minc:define(input_minc:store_dims(), minc2_file.MINC2_BYTE, minc2_file.MINC2_FLOAT)
    out_minc:create(out_file_raw)
    out_minc:setup_standard_order()
    out_minc:save_complete_volume(t_out1)
    out_minc:close()
end