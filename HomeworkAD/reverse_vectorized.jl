module VectReverse

mutable struct VectNode
    op::Union{Nothing,Symbol}
    args::Vector{VectNode}
    value::AbstractArray{Float64}     
    derivative::AbstractArray{Float64}
end

# Constructor for leaf nodes (initial values)
function VectNode(value::AbstractArray{Float64})
    return VectNode(nothing, VectNode[], value, zero(value))
end

# Constructor for operation nodes
function VectNode(op::Symbol, args::Vector{VectNode}, value::AbstractArray{Float64})
    return VectNode(op, args, value, zero(value))
end

# For `tanh.(X)`
function Base.broadcasted(op::Function, x::VectNode)
    if op !== tanh
		error("Only tanh is implemented")
	end
	value = tanh.(x.value)
	return VectNode(:tanh, [x], value)
end

# For `X .* Y`
function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    if op !== (*)
		error("Only element-wise multiplication is implemented")
	end
	value = x.value .* y.value
	return VectNode(:.*, [x, y], value)
end

# For `X .* Y` where `Y` is a constant
function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
    if op !== (*)
		error("This operation is not implemented")
	end
	if op === (^)
		value = x.value .^ y
		return VectNode(:.^, [x, y], value)
	end
	value = x.value .* y
	return VectNode(:.*, [x, y], value)
end

# For `X .* Y` where `X` is a constant
function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
    if op !== (*)
		error("Only element-wise multiplication is implemented")
	end
	value = x .* y.value
	return VectNode(:.*, [x, y], value)
end

# For `x .^ 2`
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
	Base.broadcasted(^, x, y)
end

# We assume `Flatten` has been defined in the parent module.
# If this fails, run `include("/path/to/Flatten.jl")` before
# including this file.
import ..Flatten

function backward!(f::VectNode)
	error("GLHF :)")
end


function gradient!(f, g::Flatten, x::Flatten)
	x_nodes = Flatten(VectNode.(x.components))
	expr = f(x_nodes)
	backward!(expr)
	for i in eachindex(x.components)
		g.components[i] .= x_nodes.components[i].derivative
	end
	return g
end

gradient(f, x) = gradient!(f, zero(x), x)

end


# End of part 1 : Benchmark your implementation and comment on the result, e.g., 
# What is the bottleneck in the computation of the gradient ? Does this match your expectation/complexity analysis ? 
# How would the memory and time requirement evolve with the size of the neural network ? 
# How could this be reduced ?

# ∇f = @time VectReverse.gradient(quad, x)
# @benchmark $VectReverse.gradient($L, $w)

# @profview VectReverse.gradient(L, w)

# If you see a high number of allocations, this may also be a sign of performance issue.
# You can investigate where they come from with

# @profview_allocs VectReverse.gradient(L, w)