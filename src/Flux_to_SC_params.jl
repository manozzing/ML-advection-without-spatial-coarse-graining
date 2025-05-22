#### This Julia file has functions to build the model using Flux.jl and then transfer model parameters to SimpleChains.jl.

xdim = 225
conc = rand(Float32, xdim)
cfl = rand(Float32, xdim)
input = reshape(hcat(conc, cfl), (xdim, 1, 2, 1))

function Flux_model_params(space_factor, temp_factor)
    firstlayer = 10#10
    secondlayer = 10#10
    thirdlayer = 2#3*4

    model_1x1t = Chain( Flux.Conv((3,1), 2 => firstlayer, Flux.gelu), ## 3 - 1 - 3
            Flux.Conv((3,1), firstlayer => secondlayer, Flux.gelu),
            Flux.Conv((3,1), secondlayer => thirdlayer, Flux.tanh))

    model_1x2t = Chain( Flux.Conv((5,1), 2 => firstlayer, Flux.gelu), ## 4 - 1 - 4
            Flux.Conv((3,1), firstlayer => secondlayer, Flux.gelu),
            Flux.Conv((3,1), secondlayer => thirdlayer, Flux.tanh))

    model_1x4t = Chain( Flux.Conv((5,1), 2 => firstlayer, Flux.gelu), ## 6 - 1 - 6
            Flux.Conv((5,1), firstlayer => secondlayer, Flux.gelu),
            Flux.Conv((5,1), secondlayer => thirdlayer, Flux.tanh))

    model_1x8t = Chain( Flux.Conv((7,1), 2 => firstlayer, Flux.gelu), ## 8 - 1 - 8
            Flux.Conv((7,1), firstlayer => secondlayer, Flux.gelu),
            Flux.Conv((5,1), secondlayer => thirdlayer, Flux.identity))

    MP_1x1t = Flux.MaxPool((7,), pad=3, stride=(1,))
    MP_1x2t = Flux.MaxPool((9,), pad=4, stride=(1,))
    MP_1x4t = Flux.MaxPool((13,), pad=6, stride=(1,))
    MP_1x8t = Flux.MaxPool((17,), pad=8, stride=(1,))


    factor_ratio = temp_factor÷space_factor
    if factor_ratio <= 1
        model = model_1x1t
        MP = MP_1x1t
    elseif factor_ratio == 2
        model = model_1x1t
        MP = MP_1x1t
    elseif factor_ratio == 4
        model = model_1x1t
        MP = MP_1x1t
    elseif factor_ratio == 8
        model = model_1x2t
        MP = MP_1x2t
    elseif factor_ratio == 12
        model = model_1x4t
        MP = MP_1x4t
    elseif factor_ratio == 16
        model = model_1x4t
        MP = MP_1x4t
    elseif factor_ratio == 32
        model = model_1x8t
        MP = MP_1x8t
    elseif factor_ratio == 64
        model = model_1x8t
        MP = MP_1x8t
    end
    return model, MP
end

function param_Flux_to_SC(temp_factor, ps)
    if temp_factor == 4
        ## for 4dt
        model_SC = SimpleChain(SimpleChains.Conv(SimpleChains.gelu, (3, 1), 10),
                SimpleChains.Conv(SimpleChains.gelu, (3, 1), 10),
                SimpleChains.Conv(SimpleChains.tanh, (3, 1), 2))
        p_SC = SimpleChains.init_params(model_SC, size(input))

        for i in 0:9
            for j in 0:1
            p_SC[1+6*i+3*j:3+6*i+3*j] = ps[1][3:-1:1,1,j+1,i+1]
            end
        end
        p_SC[61:70] = ps[2][1:10]
        for i in 0:9
            for j in 0:9
            p_SC[71+30*i+3*j:73+30*i+3*j] = ps[3][3:-1:1,1,j+1,i+1]
            end
        end
        p_SC[371:380] = ps[4][1:10]
        for i in 0:1
            for j in 0:9
            p_SC[381+30*i+3*j:383+30*i+3*j] = ps[5][3:-1:1,1,j+1,i+1]
            end
        end
        p_SC[441:442] = ps[6][1:2]
    elseif temp_factor == 8
        ## for 8dt
        model_SC = SimpleChain(SimpleChains.Conv(SimpleChains.gelu, (5, 1), 10),
                SimpleChains.Conv(SimpleChains.gelu, (3, 1), 10),
                SimpleChains.Conv(SimpleChains.tanh, (3, 1), 2))
        p_SC = SimpleChains.init_params(model_SC, size(input))

        for i in 0:9
            for j in 0:1
            p_SC[1+10*i+5*j:5+10*i+5*j] = ps[1][5:-1:1,1,j+1,i+1]
            end
        end
        p_SC[101:110] = ps[2][1:10]
        for i in 0:9
            for j in 0:9
            p_SC[111+30*i+3*j:113+30*i+3*j] = ps[3][3:-1:1,1,j+1,i+1]
            end
        end
        p_SC[411:420] = ps[4][1:10]
        for i in 0:1
            for j in 0:9
            p_SC[421+30*i+3*j:423+30*i+3*j] = ps[5][3:-1:1,1,j+1,i+1]
            end
        end
        p_SC[481:482] = ps[6][1:2]
    elseif 12 <= temp_factor <= 16 
        ## for 12~16dt
        model_SC = SimpleChain(SimpleChains.Conv(SimpleChains.gelu, (5, 1), 10),
                SimpleChains.Conv(SimpleChains.gelu, (5, 1), 10),
                SimpleChains.Conv(SimpleChains.tanh, (5, 1), 2))
        p_SC = SimpleChains.init_params(model_SC, size(input))

        for i in 0:9
            for j in 0:1
            p_SC[1+10*i+5*j:5+10*i+5*j] = ps[1][5:-1:1,1,j+1,i+1]
            end
        end
        p_SC[101:110] = ps[2][1:10]
        for i in 0:9
            for j in 0:9
            p_SC[111+50*i+5*j:115+50*i+5*j] = ps[3][5:-1:1,1,j+1,i+1]
            end
        end
        p_SC[611:620] = ps[4][1:10]
        for i in 0:1
            for j in 0:9
            p_SC[621+50*i+5*j:625+50*i+5*j] = ps[5][5:-1:1,1,j+1,i+1]
            end
        end
        p_SC[721:722] = ps[6][1:2]
    else
        ## for >=32dt
        model_SC = SimpleChain(SimpleChains.Conv(SimpleChains.gelu, (7, 1), 10),
                SimpleChains.Conv(SimpleChains.gelu, (7, 1), 10),
                SimpleChains.Conv(SimpleChains.identity, (5, 1), 2))
        p_SC = SimpleChains.init_params(model_SC, size(input))

        for i in 0:9
            for j in 0:1
            p_SC[1+14*i+7*j:7+14*i+7*j] = ps[1][7:-1:1,1,j+1,i+1]
            end
        end
        p_SC[141:150] = ps[2][1:10]
        for i in 0:9
            for j in 0:9
            p_SC[151+70*i+7*j:157+70*i+7*j] = ps[3][7:-1:1,1,j+1,i+1]
            end
        end
        p_SC[851:860] = ps[4][1:10]
        for i in 0:1
            for j in 0:9
            p_SC[861+50*i+5*j:865+50*i+5*j] = ps[5][5:-1:1,1,j+1,i+1]
            end
        end
        p_SC[961:962] = ps[6][1:2]
    end
    return model_SC, p_SC
end