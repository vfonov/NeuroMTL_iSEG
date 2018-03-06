-------------------------------- MNI Header -----------------------------------
-- @NAME       :  train.lua
-- @DESCRIPTION:  deep-net training
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
paths.dofile('tiles.lua')
paths.dofile('kappa.lua')
paths.dofile('data.lua')
paths.dofile('apply_fcn.lua')

require 'minc2_simple'
paths.dofile('debug.lua')


function train_fcn(opt)
    --local model          = opt.model
    local criterion      = opt.criterion
    local model_opts     = opt.model_opts
    local program        = opt.program
    local pretrain_model = opt.pretrain_model
    local train_samples  = opt.train_samples
    local validation_samples = opt.validation_samples
    local temp_dir       = opt.temp_dir
    local mean_sd        = model_opts.mean_sd
    local fold           = opt.fold
    local model_variant  = opt.model_variant
    local run_opt        = false
    local show_progress  = opt.show_progress
    local show_confusion = opt.show_confusion
    local upsampler      = nn.SpatialUpSamplingNearest(10):float()
    if opt.show_confusion == nil then show_confusion=true end

    local display_error = {
        title = string.format("Training Error, fold %d/%d",fold+1,model_opts.run_folds),
        labels = {"batch", "training"},
        ylabel = "-log10(loss)",
    }

    local error_data = {}
    
    local display_kappa = {
        title = string.format("Training Kappa, fold %d/%d",fold+1,model_opts.run_folds),
        labels = {"batch", "training","validation"},
        ylabel = "overlap",
    }
    
    local display_sample_in  = {use_png=true, title="input"}
    local display_sample_out = {use_png=true, title="output"}
    local display_debug_in   = {use_png=true, title="debug input"}
    local display_debug_seg  = {use_png=true, title="debug seg"}
    local display_debug_out  = {use_png=true, title="debug output"}
    local display_confusion  = {use_png=true}
    local display_confusion_v= {use_png=true}
    local kappa_data         = {}

    local display_lut=image.jetColormap(model_opts.classes):t()

    
    local total_batches=0
    local _,p
    for _,p in ipairs(program) do
        total_batches=total_batches+p.batches
    end
    
    
    iter_progress={}
    opt.model:training() -- enable dropout
    
    local batch=0
    
    local sampler_state={} -- state for sampler
    
    local op_config=nil
    local op_state={}
    
    local val_error=0.0
    local val_kappa=0.0
    local op=optim.adam
    
    local val_sz,val_out_cls
    if validation>0 then
        local _minc=minc2_file.new(validation_samples[1][1])
        _minc:setup_standard_order()
        val_sz=_minc:volume_size()
        val_out_cls=torch.IntTensor(val_sz)
    end

    
    local minibatch,out_size=allocate_tiles_fcn_ss(program[1].samples or 1, model_opts.border, model_opts.fov, model_opts.features) -- allocate data in GPU
    -- optimizing memory
    do
        memory_opts = {inplace=true, mode='training'}
        optnet = require 'optnet'
        optnet.optimizeMemory(opt.model, minibatch[1], memory_opts)
    end
    
    for _,p in ipairs(program) do
        local random_opt     = p.random_opt or opt.random_opt

        local parameters, gradParameters = opt.model:getParameters()
        if op_config==nil or p.reset then
            op_config = {
                learningRate = p.LR,
                learningRateDecay = p.LRd,
                momentum    = p.momentum ,
                dampening   = p.dampening,
                weightDecay = p.WD,
                beta1       = p.beta1,
                beta2       = p.beta2,
                epsilon     = p.eps,
                nesterov    = p.nesterov 
            }
            op_state = {  }
        end
        
        if p.rehash then
            sampler_state={}
        end
        
        if p.optim then op=p.optim end
        
        local j
        --xlua.progress(batch,total_batches)
        local sample_files={}
        
        for j = 1, p.batches do
            collectgarbage()
            batch=batch+1
            
            if j%p.regenerate==1 or #sample_files==0 then --
                local random_samples=generate_random_samples(train_samples, random_opt, temp_dir, model_opts)
                local i,j
                
                for i,j in ipairs(train_samples) do
                    sample_files[#sample_files+1]=j
                end
                
                -- add random samples into training dataset
                for i,j in ipairs(random_samples) do
                    sample_files[#sample_files+1]=j
                end
                sampler_state={}
            end
            
            if j%p.rehash==0 then
                sampler_state={}
            end
            
            get_samples_fcn_random_files(minibatch, sample_files,
                                         p.samples, sampler_state,
                                         model_opts.border, 
                                         model_opts.fov, 
                                         model_opts.features, 
                                         mean_sd, 
                                         model_opts.discrete_features)
            local avg_err=0
            local last_err=0
            local i
            
            for i=1,p.iter do
                local outputs
                local err
                
                local feval = function(x)
                    opt.model:zeroGradParameters()
                    outputs = opt.model:forward(minibatch[1])
                    err = criterion:forward(outputs, minibatch[2])
                    local gradOutputs = criterion:backward(outputs, minibatch[2])
                    opt.model:backward( minibatch[1], gradOutputs)
                    return err, gradParameters
                end
                
                op(feval, parameters, op_config, op_state)
                avg_err=avg_err+err
                last_err=err
            end

            local outputs=opt.model:forward(minibatch[1])
            local fov=model_opts.fov
            local border=model_opts.border
            
            local sample_in={}
            
            for i=1,model_opts.features do
                sample_in[#sample_in+1]=minibatch[1][{{1},{i},{},{fov/2},{}}]:contiguous():view(fov,fov)
            end
            
            local sample_seg = minibatch[2]:
                view(p.samples,fov-border*2,fov-border*2,fov-border*2)[{{1},{},{(fov-border*2)/2},{}}]:
                contiguous():view(fov-border*2,fov-border*2)
            
            local sample_out=outputs:
                view(p.samples,fov-border*2,fov-border*2,fov-border*2,model_opts.classes)[{{1},{},{},{},{2}}]:
                contiguous():view(fov-border*2,fov-border*2,fov-border*2)[{{},{(fov-border*2)/2},{}}]:
                contiguous():view(fov-border*2,fov-border*2)
            
            display_debug_in.win  = display.image(sample_in,              display_debug_in) 
            display_debug_seg.win = display.image(display_lut:index(2,sample_seg:long():view(-1)):
                                        view(3,sample_seg:size()[1],sample_seg:size()[2]),             display_debug_seg)  
            display_debug_out.win = display.image(sample_out,             display_debug_out)
            
            if show_confusion then
                local confusion_matrix = calc_confusion_matrix_inter_norm(outputs, minibatch[2], model_opts.classes):
                    view(1, model_opts.classes,model_opts.classes)
                
                display_confusion.title = string.format("training confusion matrix  Batch %d fold %d",batch,fold)
                display_confusion.win = display.image(torch.squeeze(upsampler:forward(confusion_matrix:float())),display_confusion)
            end
            
            avg_err=-math.log10(avg_err/p.iter)
            
            local kappa=calc_kappa_inter_gen(outputs,minibatch[2])
            
            if j%p.validate==0 and validation>0 then -- time to run validation experiment
                local avg_val_error=0.0
                local avg_val_kappa=0.0
                
                local _minibatch={ minibatch[1]:narrow(1,1,1),
                                   minibatch[2]:narrow(1,1,minibatch[2]:size(1)/p.samples) }
                
                opt.model:evaluate()
                
                local adj_fov=model_opts.fov-model_opts.border*2
                local tiles=0
                
                local dsize=model_opts.data_size
                local sample_in={}
                local v_cm=torch.zeros(model_opts.classes,model_opts.classes)
                
                for i=1,validation do
                    -- assume validation sample have the same dimension!
                    local ground_truth
                    local in_volumes=get_volumes_files(validation_samples[i], model_opts.features, mean_sd, model_opts.discrete_features)
                    
                    val_out_cls:fill(1)
                    apply_fcn({ 
                        model=opt.model,
                        fov=model_opts.fov,
                        border=model_opts.border,
                        classes=model_opts.classes,
                        features=model_opts.features, 
                        in_volumes=in_volumes,
                        minibatch=_minibatch[1],
                        out_volume=val_out_cls,
                        discrete=model_opts.discrete_features
                        })
                        
                    v_cm:add(calc_confusion_matrix_vol(val_out_cls, in_volumes[#in_volumes],model_opts.classes, model_opts.border))
                    
                    -- make it 0-based
                    val_out_cls:add(-1)
                    in_volumes[#in_volumes]:add(-1)
                    local ka=calc_kappa_gen(val_out_cls, in_volumes[#in_volumes],  model_opts.border )
                    avg_val_kappa=avg_val_kappa+ka
                    if i==validation then
                        for f=1,model_opts.features do
                            sample_in[f]=in_volumes[f][{{},{},{val_sz[3]/2}}]:contiguous():view(val_sz[1],val_sz[2])
                        end
                    end
                end 

                v_cm:cdiv(torch.sum(v_cm,2):expand(v_cm:size())) -- add epsilon?
                display_confusion_v.title = string.format("validation confusion matrix Batch %d fold %d",batch,fold)
                display_confusion_v.win = display.image(torch.squeeze(upsampler:forward(v_cm:float():view(1,model_opts.classes,model_opts.classes))),display_confusion_v)
                
                --local sample_out=val_out_cls[{{border+1,dsize[1]-border},{border+1,dsize[2]-border},{dsize[3]/2}}]:contiguous():view(dsize[1]-border*2,dsize[2]-border*2)
                
                local sample_out=val_out_cls[{{},{},{val_sz[3]/2}}]:contiguous():view(val_sz[1],val_sz[2])

                display_sample_in.title  = string.format("IN  Batch %d fold %d",batch,fold)
                display_sample_out.title = string.format("OUT Batch %d fold %d",batch,fold)

                display_sample_in.win  = display.image(sample_in, display_sample_in) 
                display_sample_out.win = display.image(
                        display_lut:index(2,sample_out:long():view(-1)):
                                        view(3,sample_out:size()[1],sample_out:size()[2])
                        ,display_sample_out)

                opt.model:training()
                val_kappa=avg_val_kappa/validation

                if val_kappa ~= val_kappa then -- check for NaN
                    val_kappa=0
                end
            end
            
            table.insert(error_data, {batch, avg_err} )
            table.insert(kappa_data, {batch, kappa,   val_kappa} )
            
            
            display_error.win = display.plot(error_data, display_error)
            display_kappa.win = display.plot(kappa_data, display_kappa)
            
            iter_progress[#iter_progress+1] = {batch=batch, iter=p.iter+1, kappa=kappa, err=avg_err, v_kappa=val_kappa,v_err=val_error}
            
            save_progress(iter_progress, model_variant..'progress.txt')
            if show_progress then
                xlua.progress(batch,total_batches)
            else
                print(string.format("%04d/%04d,%f,%f,%f",batch,total_batches,avg_err,val_error,kappa))
            end
            
            collectgarbage()
        end
    end
    opt.model:clearState()
    
    cutorch.synchronizeAll() --?
    opt.model:evaluate() -- disable dropout
    
    
    return opt.model
end
--kate: indent-width 4; replace-tabs on; show-tabs on;hl lua
