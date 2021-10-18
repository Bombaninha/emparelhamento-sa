#!/usr/bin/env julia
using JuMP
using GLPK
using GLPKMathProgInterface
using ArgParse

function parse_commandline(args)
    s = ArgParseSettings()

    @add_arg_table s begin
        "--file-path"
            help = "Path with file that contains the problem data"
            required = true
    end

    return parse_args(args, s)
end

function parse_numbers(stream)
    matches = eachmatch(r"-?\d+\.?\d*", stream)
    gen = (parse(Int, m.match) for m in matches)
    return collect(gen)
end

function main(args)
    parsed_args = parse_commandline(args)
    file_path = ""
    for (arg, val) in parsed_args
        if arg == "file-path"
            file_path = val
        end
    end
    
    n_vert = 0
    Ev = Dict{Int, Set{Int}}()
    T = Dict{Int, Set{Int}}()
    edge = 1
    open(file_path) do f
        while ! eof(f)
            stream = readline(f)
            if occursin("vértices", stream)
                n_vert = parse(Int, readline(f))
                readline(f)
                readline(f)
            else
                nums = parse_numbers(stream)
                u_set = get(Ev, nums[1], 0)
                v_set = get(Ev, nums[2], 0)
                t_set = get(T, nums[3], 0)
                if u_set != 0
                    push!(u_set, edge)
                else
                    merge!(Ev, Dict(nums[1] => Set([edge])))
                end
                if v_set != 0
                    push!(v_set, edge)
                else
                    merge!(Ev, Dict(nums[2] => Set([edge])))
                end
                if t_set != 0
                    push!(t_set, edge)
                else
                    merge!(T, Dict(nums[3] => Set([edge])))
                end
                edge += 1
            end
        end
    end
    
    edge -= 1
    println("Número de arestas: $(edge)")
    edge_count = collect(1:edge)
    m = Model()
    set_optimizer(m, GLPK.Optimizer)
    @variable(m, x[edge_count], Bin)
    @objective(m, Max, sum(x[i] for i in edge_count))
    @constraint(m, [i=keys(Ev)], sum(x[j] for j in Ev[i]) <= 1)
    @constraint(m, [i=keys(T)], sum(x[j] for j in T[i]) <= 1)
    set_time_limit_sec(m, 1800)
    println("Começando a otimização")
    optimize!(m)
    println("Valor ótimo: $(objective_value(m))")
end

main(ARGS)