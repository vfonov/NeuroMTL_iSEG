-------------------------------- MNI Header -----------------------------------
-- @NAME       :  model.lua
-- @DESCRIPTION:  creation of u-net
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
local function deeper_f(net, ops )
    --net, map_in, map_out, map_net_in, map_net_out, l1, l2, ds, bypass, up, cnt
  
    local p1=math.floor((ops.l1-1)/2)
    local p2=math.floor((ops.l2-1)/2)
    local cat
    local block=nn.Sequential()
    local contiguous=ops.contiguous
    local max_pool=ops.max_pool
    
    if contiguous == nil  then contiguous=true end
    if max_pool   == nil  then max_pool=true end
    
    local pad_up=math.ceil((ops.up-ops.ds)/2)
    local adj_up=2*pad_up-ops.up+ops.ds
    local input_layers=ops.input_layers or ops.map_in
    
    --print(string.format("deeper: ds=%d up=%d pad_up=%d adj_up=%d",ds,up,pad_up,adj_up))
    block:add(nn.VolumetricBatchNormalization(input_layers))
  
    if ops.bypass then
        cat = nn.Concat(2)
    end
    
    local sub_block=nn.Sequential()
    
    if input_layers>ops.map_in then
      sub_block:add(nn.Narrow(2,1,ops.map_in))
    end
    
    if p1>0 then sub_block:add(nn.VolumetricReplicationPadding(p1, p1, p1, p1, p1, p1)) end
    
    sub_block:add(nn.VolumetricConvolution(ops.map_in, ops.map_net_in, ops.l1, ops.l1, ops.l1, 1,1,1 ))
    sub_block:add(nn.ReLU(true))
    
    if max_pool then
      sub_block:add(nn.VolumetricMaxPooling    (ops.ds,ops.ds,ops.ds, ops.ds,ops.ds,ops.ds)) -- reduce by factor of ds
    else
      sub_block:add(nn.VolumetricAveragePooling(ops.ds,ops.ds,ops.ds, ops.ds,ops.ds,ops.ds)) -- reduce by factor of ds
    end
    
    sub_block:add(net)
    
    if contiguous then
        sub_block:add(nn.Contiguous()) -- needed for FullConvolution, when minibatch > 1
    end
    
    sub_block:add(nn.VolumetricFullConvolution(
                     ops.map_net_out,ops.map_net_out,
                     ops.up,ops.up,ops.up, 
                     ops.ds,ops.ds,ops.ds, 
                     pad_up,pad_up,pad_up, 
                     adj_up,adj_up,adj_up ))
    -- todo : add batch normalization here ?
    if contiguous then
        sub_block:add(nn.Contiguous()) -- needed for FullConvolution, when minibatch > 1
    end
    
    if ops.bypass then
        cat:add(sub_block)
        cat:add(nn.Identity())
        
        block:add(cat)
        if p2>0 then block:add(nn.VolumetricReplicationPadding(p2, p2, p2, p2, p2, p2)) end
        block:add(nn.VolumetricConvolution(input_layers+ops.map_net_out, ops.map_out, ops.l2, ops.l2, ops.l2, 1,1,1 ))
        block:add(nn.ReLU(true))
    else
        block:add(sub_block)
    end
    return block
end

local function inner_f(maps_in, maps_out, l1, l2, l3)
    local inner = nn.Sequential() -- deepest component
    inner:add(nn.VolumetricBatchNormalization(maps_in))
    local inner_concat=nn.Concat(2)
    local cnt=0
    do
      local inner_block_1=nn.Sequential()
      local inner_block_2=nn.Sequential()
      local inner_block_3=nn.Sequential()
      
      if l1~=nil and l1>0 then
        local p1=math.floor((l1-1)/2)
        if p1>0 then
          inner_block_1:add(nn.VolumetricReplicationPadding(p1, p1, p1, p1, p1, p1))
        end
        inner_block_1:add(nn.VolumetricConvolution( maps_in, maps_out, l1,l1,l1, 1,1,1))
        cnt=cnt+1
        inner_concat:add(inner_block_1)
      end
      
      if l2~=nil and l2>0 then
        local p2=math.floor((l2-1)/2)
        if p2>0 then
          inner_block_2:add(nn.VolumetricReplicationPadding(p2, p2, p2, p2, p2, p2))
        end
        inner_block_2:add(nn.VolumetricConvolution( maps_in, maps_out, l2,l2,l2, 1,1,1))
        cnt=cnt+1
        inner_concat:add(inner_block_2)
      end
      
      if l3~=nil and l3>0 then
        local p3=math.floor((l3-1)/2)
        if p3>0 then
          inner_block_3:add(nn.VolumetricReplicationPadding(p3, p3, p3, p3, p3, p3))
        end
        inner_block_3:add(nn.VolumetricConvolution( maps_in, maps_out, l3,l3,l3, 1,1,1))
        cnt=cnt+1
        inner_concat:add(inner_block_3)
      end
    end
    
    inner:add(inner_concat)
    
    inner:add(nn.ReLU(true))
    inner:add(nn.VolumetricBatchNormalization(maps_out*cnt))
    
    inner:add(nn.VolumetricConvolution( maps_out*cnt, maps_out, 1,1,1, 1,1,1))
    inner:add(nn.ReLU(true))
    
    return inner 
end  


function make_unet_mk2(opts)
  
    local fov=opts.fov
    
    -- downsamping 
    local features=opts.features or 1

    local levels=#opts.config
    -- config supposed to have
    -- map_in  - input maps
    -- map_out - output maps
    -- l1 - input layer size
    -- l2 - output layer size
    -- or , for the last layer 
    -- l1,l2,l3 - convolutions in parallel
    
    -- ds - downsampling factor [2]
    -- bypass - use bypass 
    -- up -- upsampling kernel size
    
    
    -- linear layer at highest resolution
    local hu1=opts.hu1 or 20
    local hu2=opts.hu2
    
    local outputs=opts.outputs or opts.classes or 2
    
    local border=opts.border
    local final_dropout=opts.final_dropout
    
    local data_size=(fov-border*2)*(fov-border*2)*(fov-border*2)

    -- prepare network
    
    local model = nn.Sequential()  -- make a U-net
    
    local l,o
    local u_mod=inner_f(opts.config[levels].map_in, opts.config[levels].map_out, opts.config[levels].l1, opts.config[levels].l2, opts.config[levels].l3 )
    
    for l=levels-1,1,-1 do
      u_mod=deeper_f ( u_mod, 
                          {map_in      = opts.config[l].map_in,
                           map_out     = opts.config[l].map_out,
                           map_net_in  = opts.config[l+1].map_in, 
                           map_net_out = opts.config[l+1].map_out,
                           l1          = opts.config[l].l1, 
                           l2          = opts.config[l].l2, 
                           ds          = opts.config[l].ds or 2, 
                           bypass      = opts.config[l].bypass, 
                           up          = opts.config[l].up,
                           input_layers= opts.config[l].input_layers,
                           max_pool    = opts.config[l].max_pool,
                           contiguous  = opts.config[l].contiguous
                          }
                          )
    end
    
    model:add(u_mod)
    
    
    if opts.config2 then
      local levels2=#opts.config2
      local layer_inp=opts.config[1].map_out
      for l,o in ipairs(opts.config2) do
        if o.bn~=false then
          model:add(nn.VolumetricBatchNormalization( layer_inp ))
        end
        
        if o.dropout then
          model:add(nn.VolumetricDropout( o.dropout ))
        end
        
        local layer_out=0
        local cat=nn.Concat(2)
        
        for p,k in ipairs(o) do
          local seq=nn.Sequential()
          local pad=math.floor((k.l-1)/2)
          if pad>0 then
            seq:add(nn.VolumetricReplicationPadding(pad, pad, pad, pad, pad, pad))
          end
          seq:add(nn.VolumetricConvolution( layer_inp, k.m, k.l,k.l,k.l, 1,1,1))
          layer_out=layer_out+k.m
          cat:add(seq)
        end
        model:add(cat)
        if o.relu~=false then
          model:add(nn.ReLU(true))
        end
        layer_inp=layer_out
      end
    else
      model:add(nn.VolumetricDropout( final_dropout ))
      model:add(nn.VolumetricBatchNormalization( opts.config[1].map_out ))
      model:add(nn.VolumetricConvolution( opts.config[1].map_out, hu1, 1,1,1, 1,1,1))
      model:add(nn.ReLU(true))

      -- TODO: get rid of the final RELU?
    
      if hu2 then
       model:add(nn.VolumetricBatchNormalization( hu1 ))
       model:add(nn.VolumetricConvolution( hu1, hu2, 1,1,1, 1,1,1))
       model:add(nn.ReLU(true))
     
       model:add(nn.VolumetricBatchNormalization( hu2 ))
       model:add(nn.VolumetricConvolution( hu2, outputs, 1,1,1, 1,1,1))
       model:add(nn.ReLU(true))
      else
       model:add(nn.VolumetricBatchNormalization( hu1 ))
       model:add(nn.VolumetricConvolution( hu1, outputs, 1,1,1, 1,1,1))
       model:add(nn.ReLU(true))
      end
    end
    
    if border~=nil and border>0 then
        model:add(nn.Narrow( 3, border+1, fov-border*2 ))
        model:add(nn.Narrow( 4, border+1, fov-border*2 ))
        model:add(nn.Narrow( 5, border+1, fov-border*2 ))
    end
    
    -- moving outputs into the last dimension
    model:add(nn.Transpose( {2,3},{3,4},{4,5} ))
    
    -- make a single view for cost function
    model:add(nn.View( -1, outputs ))
    
    return model
end

function make_model_unet_mk2(opts)

    local model=make_unet_mk2(opts)
    
    model:add(nn.LogSoftMax())
    
    cudnn.convert(model, cudnn, function(module)
          return torch.type(module):find( 'BatchNormalization'           ) or 
                 torch.type(module):find( 'VolumetricBatchNormalization' ) or -- disable BatchNormalization conversion for now
                 torch.type(module):find( 'VolumetricMaxPooling'         ) or
                 torch.type(module):find( 'VolumetricMaxUnpooling'       )
          end)

--    cudnn.convert(model, cudnn)

    model=model:cuda() --Half()

    local criterion = nn.ClassNLLCriterion()
    criterion = criterion:cuda()

    -- convert to half tensors
    -- model:insert(nn.Copy('torch.CudaTensor','torch.CudaHalfTensor'):cuda(),1)
    -- model:add(nn.Copy('torch.CudaHalfTensor','torch.CudaTensor'):cuda())
    
    return model,criterion
end



function adopt_model_unet_mk2(model,opts)
    -- adopt model  to have different number of input features
    local found_softmax=false
    if opts.features ~= nil and opts.features ~= model:get(1):get(1) then
        model:replace(function(module)
                if torch.typename(module) == 'nn.VolumetricBatchNormalization' and module.bias:size()[1]==1 then
                    local m=nn.VolumetricBatchNormalization(opts.features)
                    local i
                    m.bias[1]=module.bias[1]
                    m.weight[1]=m.weight[1]
                    -- not converting this to cudnn
                    return m:cuda()
                elseif torch.typename(module) == 'cudnn.VolumetricConvolution' and module.weight:size()[2]==1 then
                    local _module=cudnn.convert(module,nn)
                    
                    local m=nn.VolumetricConvolution(opts.features, _module.nOutputPlane, 
                        _module.kT,   _module.kW,   _module.kH, 
                        _module.dT,   _module.dW,   _module.dH, 
                        _module.padT, _module.padW, _module.padH)
                    
                    m.bias = _module.bias:float()
                    m.weight[     {{},{1},{},{},{}} ] = _module.weight[    {{},{1},{},{},{}} ]:float()
                    
                    m.gradBias = _module.gradBias:float()
                    m.gradWeight[ {{},{1},{},{},{}} ] = _module.gradWeight[{{},{1},{},{},{}} ]:float()
                    
                    return cudnn.convert(m,cudnn):cuda()
                else
                    return module
                end
            end)
    end

    local msize=model:size()
    if torch.typename(model:get(msize))=='cudnn.LogSoftMax' then
      msize=msize-1
      found_softmax=true
    end

    local final_view_idx=msize
    local border_idx=msize-4
    local final_lin_idx=msize-6

    -- TODO: get rid of the final RELU?
    if opts.outputs~=nil and opts.outputs ~= model:get(final_view_idx).size[2] then
      print("Changing layer:")
      print(torch.typename(model:get(final_view_idx)))
      print("Output planes:",model:get(final_view_idx).size[2])
      model:get(final_view_idx):resetSize(-1, opts.outputs)
      
      local old_lin=model:get(final_lin_idx)
      if torch.type(old_lin) ~= 'cudnn.VolumetricConvolution' then
        -- this is a new model
        old_lin=old_lin:get(1):get(1)
      end
      
      print("Changing layer:")
      print(old_lin)
      old_lin=cudnn.convert(old_lin,nn):float() -- TODO: transfer weights?
      local input_plane=old_lin.nInputPlane
      
      model:remove(final_lin_idx)
      
      local new_lin=nn.VolumetricConvolution(input_plane, opts.outputs, 1,1,1, 1,1,1)
      --TODO: transfer weights?
      model:insert(cudnn.convert(new_lin,cudnn):cuda(),final_lin_idx)
    end
    
    if opts.border~=nil and opts.border ~= model.border then
      model:get(border_idx).index=opts.border+1
      model:get(border_idx).length=model.fov-opts.border*2
      --
      model:get(border_idx+1).index=opts.border+1
      model:get(border_idx+1).length=model.fov-opts.border*2
      --
      model:get(border_idx+2).index=opts.border+1
      model:get(border_idx+2).length=model.fov-opts.border*2
      
      model.border=opts.border
    end
    
    if not found_softmax and opts.add_softmax then -- add softmax 
      model:add(cudnn.convert(nn.LogSoftMax(),cudnn):cuda())
    end
    return model
end

-- kate: space-indent on; indent-width 2; indent-mode lua;replace-tabs on;show-tabs off
