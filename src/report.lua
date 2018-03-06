-------------------------------- MNI Header -----------------------------------
-- @NAME       : report.lua
-- @DESCRIPTION: progress logs generator
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
function save_progress(prog,out)
    local f = assert(io.open(out, 'w'))
    f:write("batch,iter,kappa,error,v_kappa,v_error\n")
    local _,i
    for _,i in ipairs(prog) do
        f:write(string.format("%d,%d,%f,%f,%f,%f\n",i.batch or 0,i.iter or 0,i.kappa or 0,i.err or 0,i.v_kappa or 0,i.v_err or 0))
    end
    f:close()
end
