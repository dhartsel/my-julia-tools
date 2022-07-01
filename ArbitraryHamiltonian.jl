using Parameters
using LinearAlgebra
using OpenQuantumTools
using Markdown

@with_kw struct Transmon
    ω01 = 0.0
    α = 0.0
    N::Int = 1
    γ = 0.0
    a = [y == x+1 ? sqrt(x) : 0.0 for x in 1:N, y in 1:N]
    adag = [y == x-1 ? sqrt(x-1) : 0.0 for x in 1:N, y in 1:N]
    energy = ω01*adag*a+α*((adag*a)^2-(adag*a))
    basis = Matrix{Float64}(I,N,N)
end
@with_kw struct Qubit
    ω01 = 0.0
    γ = 0.0
    N::Int = 2
    energy = ω01/2*σz
    adag = (σx+im*σy)/2
    a = (σx-im*σy)/2
    basis = [1.0 0.0; 0.0 1.0]
end
@with_kw struct Resonator
    ω01 = 0.0
    N::Int = 1
    κ = 0.0
    a = [y == x+1 ? sqrt(x) : 0.0 for x in 1:N, y in 1:N]
    adag = [y == x-1 ? sqrt(x-1) : 0.0 for x in 1:N, y in 1:N]
    energy = ω01*adag*a
    basis = Matrix{Float64}(I,N,N)
end

##
"""
    FullDim_energy(x...)

Accepts 2-10 args of structure Transmon, Qubit, and/or Resonator. Computes the energy operators of 'each' arg in the combined basis of 'all' args.

#Examples

'''jldoctest

julia> trans = Transmon(ω01=1,N=2)
       qubit = Qubit(ω01=1)
       energy_all = FullDim_energy(trans,qubit)
2-element Array{Any,1}:
 [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
 Complex{Float64}[0.5 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -0.5 + 0.0im 0.0 + 0.0im -0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.5 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -0.0 + 0.0im 0.0 + 0.0im -0.5 + 0.0im]

julia> energy_all[1] # returns the energy operator of the Transmon in the Transmon + Qubit combined basis
4×4 Array{Float64,2}:
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

julia> energy_all[2] # returns the energy operator of the Qubit in the Transmon + Qubit combined basis
4×4 Array{Complex{Float64},2}:
 0.5+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im
 0.0+0.0im  -0.5+0.0im  0.0+0.0im  -0.0+0.0im
 0.0+0.0im   0.0+0.0im  0.5+0.0im   0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -0.5+0.0im
'''
"""
function FullDim_energy(x...)
    FullDimOp_all = []
    
    N = length(x)
    if N == 2
        push!(FullDimOp_all,kron(x[1].energy,x[2].basis))
        push!(FullDimOp_all,kron(x[1].basis,x[2].energy))
        return(FullDimOp_all)
    elseif N ==3
        push!(FullDimOp_all,kron(kron(x[1].energy,x[2].basis),x[3].basis))
        push!(FullDimOp_all,kron(kron(x[1].basis,x[2].energy),x[3].basis))
        push!(FullDimOp_all,kron(kron(x[1].basis,x[2].basis),x[3].energy))
        return(FullDimOp_all)
    elseif N == 4
        push!(FullDimOp_all,kron(kron(kron(x[1].energy,x[2].basis),x[3].basis),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].energy),x[3].basis),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].basis),x[3].energy),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].energy))
        return(FullDimOp_all)
    elseif N == 5
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].energy,x[2].basis),x[3].basis),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].energy),x[3].basis),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].energy),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].energy),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].energy))        
        return(FullDimOp_all)
    elseif N == 6
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].energy,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].energy),x[3].basis),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].energy),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].energy),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].energy),x[6].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].energy))                
        return(FullDimOp_all)
    elseif N == 7
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].energy,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].energy),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].energy),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].energy),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].energy),x[6].basis),x[7].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].energy),x[7].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].energy))                        
        return(FullDimOp_all)
    elseif N == 8
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].energy,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].energy),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].energy),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].energy),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].energy),x[6].basis),x[7].basis),x[8].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].energy),x[7].basis),x[8].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].energy),x[8].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].energy))                                
        return(FullDimOp_all)
    elseif N == 9
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].energy,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].energy),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].energy),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].energy),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].energy),x[6].basis),x[7].basis),x[8].basis),x[9].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].energy),x[7].basis),x[8].basis),x[9].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].energy),x[8].basis),x[9].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].energy),x[9].basis))                                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].energy))                                        
        return(FullDimOp_all)
    elseif N == 10
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].energy,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].energy),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].energy),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].energy),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].energy),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].energy),x[7].basis),x[8].basis),x[9].basis),x[10].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].energy),x[8].basis),x[9].basis),x[10].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].energy),x[9].basis),x[10].basis))                                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].energy),x[10].basis))                                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].energy))                                        
        return(FullDimOp_all)
    else
        print("Error. This function takes between 2 - 10 arguments of Structure Transmon, Qubit, or Resonator")
    end
end
##
"""
    FullDim_a(x...)

Accepts 2-10 args of structure Transmon, Qubit, and/or Resonator. Computes the anhilation operators of 'each' arg in the combined basis of 'all' args.

#Examples

'''jldoctest

julia> trans = Transmon(ω01=1,N=2)
       qubit = Qubit(ω01=1)
       a_all = FullDim_a(trans,qubit)
2-element Array{Any,1}:
 [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
 Complex{Float64}[0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 1.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 1.0 + 0.0im 0.0 + 0.0im]

julia> a_all[1] # returns the energy operator of the Transmon in the Transmon + Qubit combined basis
4×4 Array{Float64,2}:
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> a_all[2] # returns the energy operator of the Qubit in the Transmon + Qubit combined basis
4×4 Array{Complex{Float64},2}:
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 1.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  1.0+0.0im  0.0+0.0im
'''
"""
function FullDim_a(x...)
    FullDimOp_all = []
    
    N = length(x)
    if N == 2
        push!(FullDimOp_all,kron(x[1].a,x[2].basis))
        push!(FullDimOp_all,kron(x[1].basis,x[2].a))
        return(FullDimOp_all)
    elseif N ==3
        push!(FullDimOp_all,kron(kron(x[1].a,x[2].basis),x[3].basis))
        push!(FullDimOp_all,kron(kron(x[1].basis,x[2].a),x[3].basis))
        push!(FullDimOp_all,kron(kron(x[1].basis,x[2].basis),x[3].a))
        return(FullDimOp_all)
    elseif N == 4
        push!(FullDimOp_all,kron(kron(kron(x[1].a,x[2].basis),x[3].basis),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].a),x[3].basis),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].basis),x[3].a),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].a))
        return(FullDimOp_all)
    elseif N == 5
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].a,x[2].basis),x[3].basis),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].a),x[3].basis),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].a),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].a),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].a))        
        return(FullDimOp_all)
    elseif N == 6
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].a,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].a),x[3].basis),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].a),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].a),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].a),x[6].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].a))                
        return(FullDimOp_all)
    elseif N == 7
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].a,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].a),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].a),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].a),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].a),x[6].basis),x[7].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].a),x[7].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].a))                        
        return(FullDimOp_all)
    elseif N == 8
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].a,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].a),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].a),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].a),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].a),x[6].basis),x[7].basis),x[8].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].a),x[7].basis),x[8].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].a),x[8].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].a))                                
        return(FullDimOp_all)
    elseif N == 9
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].a,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].a),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].a),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].a),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].a),x[6].basis),x[7].basis),x[8].basis),x[9].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].a),x[7].basis),x[8].basis),x[9].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].a),x[8].basis),x[9].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].a),x[9].basis))                                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].a))                                        
        return(FullDimOp_all)
    elseif N == 10
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].a,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].a),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].a),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].a),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].a),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].a),x[7].basis),x[8].basis),x[9].basis),x[10].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].a),x[8].basis),x[9].basis),x[10].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].a),x[9].basis),x[10].basis))                                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].a),x[10].basis))                                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].a))                                        
        return(FullDimOp_all)
    else
        print("Error. This function takes between 2 - 10 arguments of Structure Transmon, Qubit, or Resonator")
    end
end
##
"""
    FullDim_adag(x...)

Accepts 2-10 args of structure Transmon, Qubit, and/or Resonator. Computes the creation operators of 'each' arg in the combined basis of 'all' args.

#Examples

'''jldoctest

julia> trans = Transmon(ω01=1,N=2)
       qubit = Qubit(ω01=1)
       adag_all = FullDim_adag(trans,qubit)
2-element Array{Any,1}:
 [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
 Complex{Float64}[0.0 + 0.0im 1.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 1.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im]

julia> adag_all[1] # returns the energy operator of the Transmon in the Transmon + Qubit combined basis
4×4 Array{Float64,2}:
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0

julia> adag_all[2] # returns the energy operator of the Qubit in the Transmon + Qubit combined basis
4×4 Array{Complex{Float64},2}:
 0.0+0.0im  1.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  1.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
'''
"""
function FullDim_adag(x...)
    FullDimOp_all = []
    
    N = length(x)
    if N == 2
        push!(FullDimOp_all,kron(x[1].adag,x[2].basis))
        push!(FullDimOp_all,kron(x[1].basis,x[2].adag))
        return(FullDimOp_all)
    elseif N ==3
        push!(FullDimOp_all,kron(kron(x[1].adag,x[2].basis),x[3].basis))
        push!(FullDimOp_all,kron(kron(x[1].basis,x[2].adag),x[3].basis))
        push!(FullDimOp_all,kron(kron(x[1].basis,x[2].basis),x[3].adag))
        return(FullDimOp_all)
    elseif N == 4
        push!(FullDimOp_all,kron(kron(kron(x[1].adag,x[2].basis),x[3].basis),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].adag),x[3].basis),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].basis),x[3].adag),x[4].basis))
        push!(FullDimOp_all,kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].adag))
        return(FullDimOp_all)
    elseif N == 5
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].adag,x[2].basis),x[3].basis),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].adag),x[3].basis),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].adag),x[4].basis),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].adag),x[5].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].adag))        
        return(FullDimOp_all)
    elseif N == 6
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].adag,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].adag),x[3].basis),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].adag),x[4].basis),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].adag),x[5].basis),x[6].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].adag),x[6].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].adag))                
        return(FullDimOp_all)
    elseif N == 7
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].adag,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].adag),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].adag),x[4].basis),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].adag),x[5].basis),x[6].basis),x[7].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].adag),x[6].basis),x[7].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].adag),x[7].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].adag))                        
        return(FullDimOp_all)
    elseif N == 8
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].adag,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].adag),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].adag),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].adag),x[5].basis),x[6].basis),x[7].basis),x[8].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].adag),x[6].basis),x[7].basis),x[8].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].adag),x[7].basis),x[8].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].adag),x[8].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].adag))                                
        return(FullDimOp_all)
    elseif N == 9
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].adag,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].adag),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].adag),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].adag),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].adag),x[6].basis),x[7].basis),x[8].basis),x[9].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].adag),x[7].basis),x[8].basis),x[9].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].adag),x[8].basis),x[9].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].adag),x[9].basis))                                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].adag))                                        
        return(FullDimOp_all)
    elseif N == 10
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].adag,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].adag),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].adag),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].adag),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].adag),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].basis))        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].adag),x[7].basis),x[8].basis),x[9].basis),x[10].basis))                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].adag),x[8].basis),x[9].basis),x[10].basis))                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].adag),x[9].basis),x[10].basis))                                
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].adag),x[10].basis))                                        
        push!(FullDimOp_all,kron(kron(kron(kron(kron(kron(kron(kron(kron(x[1].basis,x[2].basis),x[3].basis),x[4].basis),x[5].basis),x[6].basis),x[7].basis),x[8].basis),x[9].basis),x[10].adag))                                        
        return(FullDimOp_all)
    else
        print("Error. This function takes between 2 - 10 arguments of Structure Transmon, Qubit, or Resonator")
    end
end

#
"""
    Couple(x, y, g)

Returns interaction matrix between x and y, where x is a Resonator Structure and y is a Transmon or Qubit Structure, and g the coupling strength between them.
"""
function Couple(a, b, c, d, g)
    coupling_matrix = g*(a+b)*(c+d)
    return(coupling_matrix)
end