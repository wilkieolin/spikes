import torch
import numpy as np
from tqdm import tqdm
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter1d

def initialize_forward(n_signals, n_neurons):
    encoder = torch.zeros(n_signals, n_neurons,)
    torch.nn.init.normal_(encoder, mean=0.0, std=1.0)
    encoder *= 0.5
    encoder = encoder / (torch.sqrt(torch.matmul(torch.ones(n_signals,1), torch.sum(encoder**2,dim=0,keepdim=True))))
    return encoder

def initialize_recurrent(n_neurons, params):
    recurrent = -0.2*torch.rand(n_neurons,n_neurons)-params['threshold']*torch.eye(n_neurons)
    return recurrent

def compute_constants(params, device='cpu'):
    params = params.copy()

    leak_current = params['leak_current']
    leak_membrane = params['leak_membrane']
    dt = params['dt']
    alpha = (1-leak_current*dt)
    beta = (1-leak_membrane*dt)
    params['alpha'] = alpha
    params['beta'] = beta

    for key in params.keys():
        x = params[key]
        x = torch.tensor(x, device=device)
        params[key] = x

    return params


#OUTDATED
def run_EI(inputs, fwd_weights, recurrent_weights, params, learn=False):
    return
    alpha = params['alpha']
    beta = params['beta']
    sigma = params['sigma']
    threshold = params['threshold']
    if learn:
        scale_ff = params['scale_rc']
        scale_rc = params['scale_ff']
        quad_cost = params['quad_cost']
        eps_ff = params['eps_ff']
        eps_rc = params['eps_rc']
    
    n_steps = inputs.shape[0]
    n_signals = fwd_weights.shape[0]
    n_neurons = recurrent_weights.shape[0]
    
    u_forward = torch.einsum("ti,ij->tj",inputs, fwd_weights)
    
    noise = torch.zeros(n_neurons)
    reset_values = torch.zeros(n_neurons)
    
    currents = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    voltage = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    spikes = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    ifr = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    
    if learn:
        weights_ff = torch.zeros(n_steps-1, n_signals, n_neurons, dtype=torch.float)
        weights_rc = torch.zeros(n_steps-1, n_neurons, n_neurons, dtype=torch.float)
    
    for i in range(n_steps-1):
        #compute the current coming into the cells
        u_recurrent = torch.matmul(spikes[i], recurrent_weights)
        currents[i+1] = u_recurrent + u_forward[i+1] + currents[i]*alpha
        
        #compute the cell voltage
        if sigma > 0.0:
            torch.nn.init.normal_(noise, mean=0.0, std=sigma)
        voltage[i+1,:] = -1*beta*voltage[i,:] + currents[i+1,:] + noise
        #voltage[i+1,:] = voltage[i+1] - voltage[i+1]*(spikes[i])
        
        #mark which cells have fired
        spikes[i+1,:] = 1.0*(voltage[i+1,:] > threshold)
        
        #update the filtered trains
        ifr[i+1,:] = (beta)*ifr[i,:] + spikes[i,:]
        
        if learn:
            spiking_neurons = (spikes[i+1,:] == 1.0)
            
            #learn on the feedforward connections
            weights_ff[i,...] = fwd_weights
            scaled_input = scale_ff*inputs[i,:].view(-1,1).repeat_interleave(n_neurons,dim=1)
            fwd_weights[:,spiking_neurons] += eps_ff * (scaled_input[:,spiking_neurons] - fwd_weights[:,spiking_neurons])
            
            
            #learn on the recurrent connections
            weights_rc[i,...] = recurrent_weights
            #pre neuron voltage + ifr regularization term, same across row
            scaled_voltage = scale_rc * voltage[i+1,:].view(n_neurons,1).repeat_interleave(n_neurons,dim=1)
            scaled_voltage += quad_cost * ifr[i+1,:].view(n_neurons,1).repeat_interleave(n_neurons,dim=1)
            post_spks = spiking_neurons.view(n_neurons,1).repeat_interleave(n_neurons,dim=1).transpose(0,1)
            quad_cost = quad_cost * torch.eye(n_neurons)
                                        
            recurrent_weights -= post_spks * eps_rc * (scaled_voltage + recurrent_weights + quad_cost)
        
    if learn:
        return currents, voltage, spikes, ifr, weights_ff, weights_rc
    else:
        return currents, voltage, spikes, ifr

#OUTDATED
def run_EI_learn(inputs, fwd_weights, recurrent_weights, params):
    return
    #cell and network parameters
    dt = params['dt']
    leak_current = params['leak_current']
    leak_membrane = params['leak_membrane']
    alpha = (1-leak_current*dt)
    beta = (1-leak_membrane*dt)
    
    sigma = params['sigma']
    threshold = params['threshold']
    scale_ff = params['scale_rc']
    scale_rc = params['scale_ff']
    quad_cost = params['quad_cost']
    eps_ff = params['eps_ff']
    eps_rc = params['eps_rc']
    
    #don't change initial weights
    fwd_weights = fwd_weights.clone()
    recurrent_weights = recurrent_weights.clone()
    
    #runtime parameters
    n_steps = inputs.shape[0]
    n_signals = fwd_weights.shape[0]
    n_neurons = recurrent_weights.shape[0]
    
    #tensors which change without memory
    noise = torch.zeros(n_neurons)
    reset_values = torch.zeros(n_neurons)
    filtered_input = torch.zeros(n_signals)
    
    #tensors tracking variables
    currents = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    voltage = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    spikes = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    ifr = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    weights_ff = torch.zeros(n_steps-1, n_signals, n_neurons, dtype=torch.float)
    weights_rc = torch.zeros(n_steps-1, n_neurons, n_neurons, dtype=torch.float)
    
    #run the network
    for i in tqdm(range(n_steps-1)):
        #update the filtered input
        filtered_input = beta*filtered_input + dt*inputs[i]
        #compute the current coming into the cells
        u_recurrent = torch.matmul(spikes[i], recurrent_weights)
        u_forward = torch.matmul(dt*inputs[i], fwd_weights)
        currents[i+1] = u_recurrent + u_forward + currents[i]*alpha
        
        #compute the cell voltage
        if sigma > 0.0:
            torch.nn.init.normal_(noise, mean=0.0, std=sigma)
        voltage[i+1,:] = beta*voltage[i,:] + currents[i+1,:] + noise
        #reset cells which fired last round #now done throuch RC conn
        #voltage[i+1,:] -= voltage[i+1]*spikes[i]
        
        #mark which cells have fired
        spikes[i+1,:] = 1.0*(voltage[i+1,:] > threshold)
        spiking_neurons = (spikes[i+1,:] >= 1.0)
        
        #update the filtered trains
        ifr[i+1,:] = (beta)*ifr[i,:] + spikes[i+1,:]
        
        #learn on the feedforward connections
        weights_ff[i,...] = fwd_weights
        scaled_input = scale_ff*inputs[i+1,:].view(-1,1).repeat_interleave(n_neurons,dim=1)
        fwd_weights[:,spiking_neurons] += eps_ff * (scaled_input[:,spiking_neurons] - fwd_weights[:,spiking_neurons])

        #learn on the recurrent connections
        weights_rc[i,...] = recurrent_weights
        #pre neuron voltage + ifr regularization term, same across row
        scaled_voltage = scale_rc * voltage[i+1,:].view(n_neurons,1).repeat_interleave(n_neurons,dim=1)
        scaled_voltage += quad_cost * ifr[i+1,:].view(n_neurons,1).repeat_interleave(n_neurons,dim=1)
        post_spks = spiking_neurons.view(n_neurons,1).repeat_interleave(n_neurons,dim=1).transpose(0,1)
        quad_cost = quad_cost * torch.eye(n_neurons)

        recurrent_weights -= post_spks * eps_rc * (scaled_voltage + recurrent_weights + quad_cost)
        
    return currents, voltage, spikes, ifr, weights_ff, weights_rc

def run_EI_onespk(inputs, fwd_weights, recurrent_weights, params):
    #cell and network parameters
    dt = params['dt']
    leak_current = params['leak_current']
    leak_membrane = params['leak_membrane']
    alpha = (1-leak_current*dt)
    beta = (1-leak_membrane*dt)
    
    sigma = params['sigma']
    threshold = params['threshold']
    
    #don't change initial weights
    fwd_weights = fwd_weights.clone()
    recurrent_weights = recurrent_weights.clone()
    
    #runtime parameters
    n_steps = inputs.shape[0]
    n_signals = fwd_weights.shape[0]
    n_neurons = recurrent_weights.shape[0]
    
    #tensors which change without memory
    noise = torch.zeros(n_neurons)
    reset_values = torch.zeros(n_neurons)
    filtered_input = torch.zeros(n_signals)
    index = 0
    
    #tensors tracking variables
    currents = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    voltage = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    spikes = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    ifr = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    
    #run the network
    for i in tqdm(range(n_steps-1)):
        #update the filtered input
        filtered_input = beta*filtered_input + dt*inputs[i]
        
        #compute the current coming into the cells
        u_recurrent = torch.matmul(spikes[i], recurrent_weights)
        u_forward = torch.matmul(dt*inputs[i], fwd_weights)
        currents[i+1] = u_recurrent + u_forward + currents[i]*alpha
        
        #compute the cell voltage
        if sigma > 0.0:
            torch.nn.init.normal_(noise, mean=0.0, std=sigma)
        voltage[i+1,:] = beta*voltage[i,:] + currents[i+1,:] + noise

        #mark which cells have fired
        vmax = torch.max(voltage[i+1] - threshold, dim=0)
        if vmax.values.item() > 0.0:
            index = vmax.indices.item()
            spikes[i+1,index] = 1.0
        else:
            index = -1
        
        #update the filtered trains
        ifr[i+1,:] = (beta)*ifr[i,:] + spikes[i+1,:]
        
    return currents, voltage, spikes, ifr

def run_EI_learn_onespk(inputs, fwd_weights, recurrent_weights, params):
    #cell and network parameters
    dt = params['dt']
    leak_current = params['leak_current']
    leak_membrane = params['leak_membrane']
    alpha = (1-leak_current*dt)
    beta = (1-leak_membrane*dt)
    
    sigma = params['sigma']
    threshold = params['threshold']
    scale_ff = params['scale_rc']
    scale_rc = params['scale_ff']
    quad_cost = params['quad_cost']
    eps_ff = params['eps_ff']
    eps_rc = params['eps_rc']
    
    #don't change initial weights
    fwd_weights = fwd_weights.clone()
    recurrent_weights = recurrent_weights.clone()
    
    #runtime parameters
    n_steps = inputs.shape[0]
    n_signals = fwd_weights.shape[0]
    n_neurons = recurrent_weights.shape[0]
    
    #tensors which change without memory
    noise = torch.zeros(n_neurons)
    reset_values = torch.zeros(n_neurons)
    filtered_input = torch.zeros(n_signals)
    index = 0
    
    #tensors tracking variables
    currents = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    voltage = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    spikes = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    ifr = torch.zeros(n_steps, n_neurons, dtype=torch.float)
    weights_ff = torch.zeros(n_steps-1, n_signals, n_neurons, dtype=torch.float)
    weights_rc = torch.zeros(n_steps-1, n_neurons, n_neurons, dtype=torch.float)
    
    #run the network
    for i in tqdm(range(n_steps-1)):
        #update the filtered input
        filtered_input = beta*filtered_input + dt*inputs[i]
        
        #compute the current coming into the cells
        u_recurrent = torch.matmul(spikes[i], recurrent_weights)
        u_forward = torch.matmul(dt*inputs[i], fwd_weights)
        currents[i+1] = u_recurrent + u_forward + currents[i]*alpha
        
        #compute the cell voltage
        if sigma > 0.0:
            torch.nn.init.normal_(noise, mean=0.0, std=sigma)
        voltage[i+1,:] = beta*voltage[i,:] + currents[i+1,:] + noise

        #mark which cells have fired
        vmax = torch.max(voltage[i+1] - threshold, dim=0)
        if vmax.values.item() > 0.0:
            index = vmax.indices.item()
            spikes[i+1,index] = 1.0
        else:
            index = -1
        
        #update the filtered trains
        ifr[i+1,:] = (beta)*ifr[i,:] + spikes[i+1,:]
        
        #learn on the feedforward connections
        weights_ff[i,...] = fwd_weights
        scaled_input = scale_ff*filtered_input
        fwd_weights[:,index] += eps_ff * (scaled_input - fwd_weights[:,index])

        #learn on the recurrent connections
        weights_rc[i,...] = recurrent_weights
        if index >= 0:
            scaled_voltage = scale_rc * (voltage[i+1,:] + quad_cost*ifr[i+1,:])
            recurrent_weights[:,index] -= eps_rc * (scaled_voltage + recurrent_weights[:,index] + quad_cost * torch.eye(n_neurons)[:,index])
        
    return currents, voltage, spikes, ifr, weights_ff, weights_rc

def generate_signal(n_time, n_dimensions, params):
    sigma = float(params['x_sigma'])
    amplitude = float(params['x_amplitude'])

    generator = st.norm()
    data = amplitude*generator.rvs(n_time*n_dimensions).reshape(n_time, n_dimensions)
    for i in range(n_dimensions):
        data[:,i] = gaussian_filter1d(data[:,i], sigma)
        
    return torch.tensor(data, dtype=torch.float)

def leaky_integrate(signal, params):
    n_time = signal.shape[0]
    n_signals = signal.shape[1]
    beta = params['beta']

    output = torch.zeros(n_time, n_signals)
    for i in range(n_time-1):
        output[i+1,:] = beta*output[i,:] + signal[i,:]
    
    return output

def compute_decoders(signal, weights_ff, weights_rc, params):
    n_signals = signal.shape[1]
    n_weights = weights_rc.shape[0]
    n_neurons = weights_rc.shape[1]

    decoders = torch.zeros(n_weights, n_neurons, n_signals)
    filtered_inputs = leaky_integrate(signal, params)
    for i in range(n_weights):
        (u,v,s,r) = run_EI_onespk(signal, weights_ff[i,...], weights_rc[i,...], params)
        decoders[i,...] = torch.matmul(torch.pinverse(r), filtered_inputs)

    return decoders

def learn_simple(signal, fwd, rc, params, device="cpu"):
    signal = signal.to(device)
    params = compute_constants(params, device)
    n_time = signal.shape[0]
    n_samples = int(np.floor(np.log(n_time)/np.log(2)))+1
    n_signals = signal.shape[1]
    n_neurons = rc.shape[0]

    beta = params['beta']
    dt = params['dt']
    threshold = params['threshold']
    scale_ff = params['scale_rc']
    scale_rc = params['scale_ff']
    quad_cost = params['quad_cost']
    eps_ff = params['eps_ff']
    eps_rc = params['eps_rc']

    weights_fwd = torch.zeros(n_samples, n_signals, n_neurons, device=device)
    weights_rc = torch.zeros(n_samples, n_neurons, n_neurons, device=device)
    exp_step = 0

    fwd = fwd.clone().to(device)
    rc = rc.clone().to(device)

    voltage = torch.zeros(n_neurons, device=device)
    leaky_input = torch.zeros(n_signals, device=device)
    noise = torch.zeros(n_neurons, device=device)
    rates = torch.zeros(n_neurons, device=device)
    spk = torch.tensor(0, dtype=torch.float, device=device)
    ind = torch.tensor(1, dtype=torch.long, device=device)

    for i in tqdm(range(n_time-1)):
        #store the weights on an exponential time scale
        if (i % 2**exp_step) == 0:
            #print("Exp step", exp_step)
            weights_fwd[exp_step,...] = fwd
            weights_rc[exp_step,...] = rc
            exp_step += 1

        #updates values
        voltage = beta*voltage + dt*torch.matmul(fwd.transpose(1,0), signal[i,:]) + spk*rc[:,ind] + 0.001*torch.nn.init.normal_(noise, mean=0.0, std=1.0)
        leaky_input = beta*leaky_input + dt*signal[i,:]

        centered = voltage - threshold - 0.01*torch.nn.init.normal_(noise, mean=0.0, std=1.0)
        info = torch.max(centered, dim=0)

        if (info.values >= 0.0):
            spk = 1.0
            ind = info.indices
            

            #do the learning
            fwd[:,ind] = fwd[:,ind] + eps_ff*(scale_ff*leaky_input - fwd[:,ind])
            rc[:,ind] = rc[:,ind] - eps_rc*(scale_rc*(voltage + quad_cost*rates) + rc[:,ind] + quad_cost*torch.eye(n_neurons, device=device)[:,ind])
            rates[ind] += 1.0
        else:
            spk = 0.0

        rates = beta*rates

    return fwd.cpu(), rc.cpu(), weights_fwd.cpu(), weights_rc.cpu()

def run_simple(signal, fwd, rc, params, device="cpu"):
    signal = signal.to(device)
    params = compute_constants(params, device)
    n_time = signal.shape[0]
    n_signals = signal.shape[1]
    n_neurons = rc.shape[0]

    beta = params['beta']
    dt = params['dt']
    threshold = params['threshold']

    
    noise = torch.zeros(n_neurons, device=device)

    voltage = torch.zeros(n_time, n_neurons, device=device)
    rates = torch.zeros(n_time, n_neurons, device=device)
    spikes = torch.zeros(n_time, n_neurons, device=device)

    spk = torch.tensor(0, dtype=torch.float, device=device)
    ind = torch.tensor(1, dtype=torch.long, device=device)

    for i in tqdm(range(n_time-1)):

        #updates values
        voltage[i+1, :] = beta*voltage[i,:] + dt*torch.matmul(fwd.transpose(1,0), signal[i,:]) + torch.matmul(rc, spikes[i,:]) + 0.001*torch.nn.init.normal_(noise, mean=0.0, std=1.0)

        centered = voltage[i+1,:] - threshold - 0.01*torch.nn.init.normal_(noise, mean=0.0, std=1.0)
        info = torch.max(centered, dim=0)

        if (info.values > 0.0):
            spk = 1.0
            ind = info.indices
            spikes[i+1, ind] = 1.0

        else:
            spk = 0.0

        rates[i+1,:] = beta*rates[i,:] + spikes[i+1,:]

    return voltage, spikes, rates

def compute_decoders_simple(signal, weights_ff, weights_rc, params):
    n_signals = signal.shape[1]
    n_weights = weights_rc.shape[0]
    n_neurons = weights_rc.shape[1]

    decoders = torch.zeros(n_weights, n_neurons, n_signals)
    filtered_inputs = leaky_integrate(signal, params)
    for i in range(n_weights):
        (v,s,r) = run_simple(signal, weights_ff[i,...], weights_rc[i,...], params)
        decoders[i,...] = torch.matmul(torch.pinverse(r), filtered_inputs)

    return decoders

def learning_metrics(n_time, n_trials, ff_weights, rc_weights, decoders, params):
    dt = params['dt']
    
    n_networks = ff_weights.shape[0]
    n_signals = ff_weights.shape[1]
    n_neurons = ff_weights.shape[2]
    
    errors = torch.zeros(n_networks, n_trials)
    mean_rate = torch.zeros(n_networks, n_trials)
    membrane_var = torch.zeros(n_networks, n_trials)
    
    
    for i in range(n_trials):
        signal = generate_signal(n_time, n_signals, params)
        leaky_signal = leaky_integrate(signal, params)
        
        for j in range(n_networks):
            v,s,r = run_simple(signal, ff_weights[j,...], rc_weights[j,...], params)
            signal_est = torch.matmul(r, decoders[j,...])
            
            errors[j,i] = torch.sum(torch.var(signal - signal_est, 0))/torch.sum(torch.var(signal, 0))
            mean_rate[j,i] = torch.sum(s)/(dt*n_time*n_neurons)
            membrane_var[j,i] = torch.sum(torch.var(v, 0))/(n_neurons)
            
    return errors, mean_rate, membrane_var