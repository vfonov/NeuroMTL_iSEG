-------------------------------- MNI Header -----------------------------------
-- @NAME       :  validate.lua
-- @DESCRIPTION:  Validation code
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


function validate(validate_opt)
    local test_samples=validate_opt.test_samples
    local model_opts=validate_opt.model_opts
    local model_variant=validate_opt.model_variant
    local lm=validate_opt.lm
    local mean_sd=validate_opt.mean_sd
    local fold=validate_opt.fold
    local model=validate_opt.model
    local criterion=validate_opt.criterion
    local discrete_features=model_opts.discrete_features
    
    local err1=0.0
    
    local val_sz
    local reference=minc2_file.new(test_samples[1][1])
    reference:setup_standard_order()
    val_sz=reference:volume_size()
    local dsize=val_sz    
    
    local t_out1=torch.FloatTensor(val_sz):fill(0)
    local t_out2=torch.IntTensor(val_sz):fill(0)

    local minibatch=allocate_tiles_fcn_ss(1, model_opts.border, model_opts.fov, model_opts.features) -- allocate data in GPU

    print("Evaluating")
    local f = assert(io.open(model_variant..'similarity.csv', 'w'))
    
    local hdr=string.format('kappa_%s',lm['1'])
    for j=2,(model_opts.classes-1) do
        hdr=hdr..string.format(',kappa_%s',lm[j..''])
    end
    
    --print(string.format("Mean=%f sd=%f",mean_sd.mean[1],mean_sd.sd[1]))
    f:write("subject,fold,error,gen_kappa," .. hdr .. "\n")

    xlua.progress(0,#test_samples)
    local j
    local eval_data={}
    for j=1,#test_samples do
        collectgarbage()
        local err1=0.0
        
        local k,l,m
        local tiles=0
        local adj_fov=model_opts.fov-model_opts.border*2
        -- go over all tiles
        local in_volumes=get_volumes_files(test_samples[j],model_opts.features, mean_sd, model_opts.discrete_features)
        t_out2:fill(1)
        apply_fcn({ 
            model=model,
            fov=model_opts.fov,
            border=model_opts.border,
            classes=model_opts.classes,
            features=model_opts.features, 
            in_volumes=in_volumes,
            minibatch=minibatch[1],
            out_volume=t_out2
            })
        -- make it 0-based
        t_out2:add(-1)
        in_volumes[#in_volumes]:add(-1)
        
        local g_kappa = calc_kappa_gen(t_out2,   in_volumes[#in_volumes], model_opts.border )
        local m_kappa = calc_kappa_multi(t_out2, in_volumes[#in_volumes], model_opts.classes, model_opts.border )
        
        local s_kappa=string.format("%f",m_kappa[1])
        local k
        for k=2,(model_opts.classes-1) do
            s_kappa=s_kappa..string.format(",%f",m_kappa[k])
        end
        
        f:write( string.format("%s,%d,%f,%f,%s\n", test_samples[j][model_opts.features+1], fold+1, err1/tiles, g_kappa, s_kappa ))
        table.insert(eval_data, {j,g_kappa} )
--          
        local out_minc=minc2_file.new()
        out_minc:define(reference:store_dims(), minc2_file.MINC2_BYTE,  minc2_file.MINC2_BYTE)
        out_minc:create(model_variant..string.format("output_%03d_s.mnc",j))
        out_minc:setup_standard_order()
        out_minc:save_complete_volume(t_out2)
        
--         local out_minc2=minc2_file.new()
--         out_minc2:define(reference:store_dims(), minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
--         out_minc2:create(model_variant..string.format("output_%03d_raw.mnc",j))
--         out_minc2:setup_standard_order()
--         out_minc2:save_complete_volume(t_out1)
    -- 
        xlua.progress(j,#test_samples)
    end
    
    display.plot(eval_data, {
        title = string.format("Evaluation kappa, fold %d/%d",fold+1,model_opts.run_folds),
        labels = {"Sample","Kappa"},
        ylabel = "overlap",
    })
end
