-------------------------------- MNI Header -----------------------------------
-- @NAME       :  dpt_util.lua
-- @DESCRIPTION:  data-parallel processing
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
require 'cunn'
require 'cudnn'


function makeDataParallel(model, nGPU)   

   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      
      local gpu_table = torch.range(1, nGPU):totable()
      local dpt = nn.DataParallelTable(1, true, true):add(model, gpu_table):threads(function() require 'cudnn'
                                       cudnn.benchmark = true  end)
      dpt.gradInput = nil
      model = dpt:cuda()
   else
       cudnn.benchmark = true
   end

   return model
end

function cleanDataParallel(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   if torch.type(module) == 'nn.DataParallelTable' then
       return module:get(1)
   else 
       -- TODO: error?
       return module
   end
end

