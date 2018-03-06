-------------------------------- MNI Header -----------------------------------
-- @NAME       :  kappa.lua
-- @DESCRIPTION:  calculating kappa and generalized kappa overlap metric
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

function calc_kappa(a,b,border)
  local border=border or 0
  local sz=a:size()
  local _a=a[{{border+1,sz[1]-border-1},{border+1,sz[2]-border-1},{border+1,sz[3]-border-1}}]:clone():add(-1)
  local _b=b[{{border+1,sz[1]-border-1},{border+1,sz[2]-border-1},{border+1,sz[3]-border-1}}]:clone():add(-1)
  local v=(torch.sum(_a)+torch.sum(_b))
  
  if v>0 then
    return 2*torch.sum( torch.cmul(_a,_b) ) / (torch.sum(_a)+torch.sum(_b))
  else
    return 0.0
  end
end

function calc_kappa_inter(a,b)
    -- calculate intermediary kappa
    --TODO make it executable on GPU?
    local _,_a=a:max(2) -- last dimension contains the outputs
    local _b=b:byte():clone():add(-1)
    _a=_a:byte():clone():add(-1)
    
    local v=(torch.sum(_a)+torch.sum(_b))
    
    if v>0 then
        return 2.0*torch.sum(torch.cmul(_a,_b) ) / v
    else 
        return 0.0
    end
end


function calc_kappa_gen(a,b,border)
    --[[
        calculates multiple volume similarity metrics for discrete labels 
or Generalized Tanimoto coefficient (GTC)
based on :  William R. Crum, Oscar Camara, and Derek L. G. Hill"Generalized Overlap Measures for Evaluation and Validation in Medical Image Analysis " IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 25, NO. 11, NOVEMBER 2006
http://dx.doi.org/10.1109/TMI.2006.880587
        --]]
    local border=border or 0
    local sz=a:size()

    local _a=a[{{border+1,sz[1]-border-1},{border+1,sz[2]-border-1},{border+1,sz[3]-border-1}}]
    local _b=b[{{border+1,sz[1]-border-1},{border+1,sz[2]-border-1},{border+1,sz[3]-border-1}}]
    
    local total_volume=torch.sum(torch.gt(_a,0))+torch.sum(torch.gt(_b,0))
    local intersect=torch.sum(torch.cmul(torch.cmul(torch.gt(_a,0),torch.gt(_b,0)),torch.eq(_a,_b)))
    
    if total_volume>0 then
        return 2*intersect/total_volume
    else
        return 0.0
    end
end

function calc_kappa_inter_gen(a,b)
    -- calculate intermediary kappa
    --TODO make it executable on GPU?
    local _,_a=a:max(2) -- last dimension contains the outputs
    local _b=b:byte():clone():add(-1)
    _a=_a:byte():add(-1)
    
    local total_volume=torch.sum(torch.gt(_a,0))+torch.sum(torch.gt(_b,0))
    local intersect=torch.sum(torch.cmul(torch.cmul(torch.gt(_a,0),torch.gt(_b,0)),torch.eq(_a,_b)))
    
    if total_volume>0.0 then
        return 2*intersect/total_volume
    else
        return 0.0
    end
end


function calc_kappa_multi(a,b,n,border)
    local border=border or 0
    local sz=a:size()
    
    local _a = a[{{border+1,sz[1]-border}, {border+1,sz[2]-border}, {border+1,sz[3]-border}}]
    local _b = b[{{border+1,sz[1]-border}, {border+1,sz[2]-border}, {border+1,sz[3]-border}}]
    
    local r={}
    for c=1,(n-1) do
        local _a_c=torch.eq(_a,c)
        local _b_c=torch.eq(_b,c)
        local total_volume=torch.sum(_a_c)+torch.sum(_b_c)
        local intersect=torch.sum(torch.cmul( _a_c, _b_c))
        r[c]=2*intersect/total_volume
    end
    r[0]=calc_kappa_gen(a,b)
    return r
end


function calc_kappa_inter_multi(a,b,n)
    -- calculate intermediary kappa
    --TODO make it executable on GPU?
    local _,_a=a:max(2) -- last dimension contains the outputs
    local _b=b:byte():clone():add(-1)
    _a=_a:byte():clone():add(-1)
    return calc_kappa_multi(_a,_b,n)
end



function calc_confusion_matrix(a, b, n)
    local _eye=torch.eye(n)
    
    local _a_=_eye:index(1, a)
    local _bt=_eye:index(2, b)
    
    return torch.mm(_bt,_a_)
end


function calc_confusion_matrix_vol(a, b, n, border)
    local border=border or 0
    local sz=torch.LongTensor(a:size())
    
    local _sz=torch.add(sz,-2*border)
    
    local _a = a[{{border+1,sz[1]-border}, {border+1,sz[2]-border}, {border+1,sz[3]-border}}]:contiguous():view(torch.prod(_sz))
    local _b = b[{{border+1,sz[1]-border}, {border+1,sz[2]-border}, {border+1,sz[3]-border}}]:contiguous():view(torch.prod(_sz))
    
    return calc_confusion_matrix(_a:long(),_b:long(),n)
end


function calc_confusion_matrix_inter(a,b,n)
    local _,_a=a:max(2) -- last dimension contains the outputs
    return calc_confusion_matrix(_a:viewAs(b):long(),b:long(),n)
end

function calc_confusion_matrix_inter_norm(a,b,n)
    --TODO: verify the dimensions order 
    local cmat=calc_confusion_matrix_inter(a,b,n)
    local sums=torch.sum(cmat,2):expand(cmat:size()):add(1e-7)
    return cmat:cdiv(sums)
end

