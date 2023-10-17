import numpy as np
from brian2 import *
prefs.codegen.target = 'numpy'

def get_BC_kernel(kernel, BC_position, extension_right, extension_left, spatial_conversion):
    """
    Extracts the kernel for a BC specific to it's location along the SAC dendrite.

    Parameters
    ----------
    kernel : numpy.ndarray
        Complete BC kernel
    BC_position : int/float
        Position of BC along the SAC dendrite in um from soma
    extension_right : int/float
        How far the stimulus moves beyond the end of the dendrite in um
    extension_left : int/float
        How far the stimulus moves along the other side of the SAC in um
    spatial_conversion : int/float
        Ratio to convert from um to pixel

    Returns
    -------
    kernel_positioned : numpy.ndarray
        Partial kernel in correct size and specific to it's location along the SAC dendrite
    """

    assert kernel.shape[1] == 1625, 'Spatial dimension wrong'
    total_distance = 150 + extension_right + extension_left # Dendrite is 150; total distance covered by stimulus in um
    total_nb_pixel = int(np.round(total_distance*spatial_conversion))
    BC_position_px = int(np.round((BC_position+extension_left)*spatial_conversion)) # Nb of pixel to the left of the BC center
    nb_pixel_right = total_nb_pixel - BC_position_px # Nb of pixel to the right of the BC center
    start = 812 - BC_position_px # All kernels are centered at 812 (half of spatial dimension)
    stop = 812 + nb_pixel_right
    kernel_positioned = kernel[:,start:stop]
    assert kernel_positioned.shape[1] == total_nb_pixel

    return kernel_positioned

def create_BC_input_array(BC_signal_outward, BC_signal_inward, synapse_locations, spatial_conversion):
    """
    Creates a 2-D input array for SAC simulation for outward and inward stimulation for one BC

    Parameters
    ----------
    BC_signal_outward : numpy.ndarray
        Time array of input for current BC for outward stimulus
    BC_signal_inward : numpy.ndarray
        Time array of input for current BC for inward stimulus
    synapse_locations : numpy.ndarray
        Contains synapse locations in um for current BC
    spatial_conversion : int/float
        Ratio to convert from um to pixel

    Returns
    -------
    outward_array, inward_array : numpy.ndarray
        time x space in pixel space, containing simulation input
    """

    assert BC_signal_outward.shape == BC_signal_inward.shape
    time_dimension = BC_signal_outward.shape[0]
    space_dimension = int(np.round(150*spatial_conversion)) + 1 # Plus one is for soma
    outward_array = np.zeros((time_dimension, space_dimension))
    inward_array = np.zeros((time_dimension, space_dimension))

    for current_synapse_location in synapse_locations:

        current_synapse_index = int(np.round(current_synapse_location*spatial_conversion)) + 1 # Plus one for soma
        outward_array[:,current_synapse_index] = BC_signal_outward
        inward_array[:,current_synapse_index] = BC_signal_inward

    return outward_array, inward_array

def get_baseline_activity(synapse_locations, spatial_conversion, baseline_activity):
    """
    Makes an array which contains the baseline activity for the BC at it's synapse locations

    Parameters
    ----------
    synapse_locations : numpy.ndarray
        Contains synapse locations of current BC in um
    spatial_conversion : int/float
        Ratio to convert from um to pixel
    baseline_activity : int/float
        Spontanous input without stimulus

    Returns
    -------
    baseline_activity_array : numpy.ndarray
        Contains baseline activity at synapse locations
    """

    nb_pixel = int(np.round(150*spatial_conversion)) + 1 # + 1 for soma, dendrite is 150 um
    baseline_activity_array = np.zeros(nb_pixel)

    for current_synapse_location in synapse_locations:

        current_synapse_index = int(np.round(current_synapse_location*spatial_conversion)) + 1 # Plus one for soma
        baseline_activity_array[current_synapse_index] = baseline_activity

    return baseline_activity_array

def perform_convolution(outward_stimulus, inward_stimulus, BC_kernel, baseline_activity,
                        response_scaling, clipping):
    """
    Performs convolution to get response of BC to outward and inward motion stimuli

    Parameters
    ----------
    outward_stimulus : numpy.ndarray
        Stimulus picture for outward motion
    inward_stimulus : numpy.ndarray
        Stimulus picture for inward motion
    BC_kernel : numpy.ndarray
        Kernel used for convolution
    baseline_activity : float/int
        Spontanous input without stimulus, gets added to result of convolution
    response_scaling : float/int
        Scaling to go from filter activity to ampere
    clipping : boolean
        Whether or not to clip the signal so that input cannot go below 0

    Returns
    -------
    outward_response_time, inward_response_time : numpy.ndarray
        Activity across time of kernel for outward and inward stimulus
    """

    assert outward_stimulus.shape == inward_stimulus.shape
    assert BC_kernel.shape[1] == outward_stimulus.shape[1], 'Spatial dimensions do not match'
    nb_points_space = BC_kernel.shape[1]
    nb_points_time = BC_kernel.shape[0] + outward_stimulus.shape[0] - 1
    response_outward = np.zeros((nb_points_time, nb_points_space))
    response_inward = np.zeros((nb_points_time, nb_points_space))

    for current_position in range(nb_points_space):
        current_kernel = BC_kernel[:,current_position]

        current_image = outward_stimulus[:,current_position]
        response_outward[:,current_position] =  np.convolve(a = current_image, v = current_kernel, mode = 'full')

        current_image = inward_stimulus[:,current_position]
        response_inward[:,current_position] =  np.convolve(a = current_image, v = current_kernel, mode = 'full')

    # Sum across space and scale arbitrary filter unit to ampere
    outward_response_time = np.sum(response_outward, axis = 1)*response_scaling
    inward_response_time = np.sum(response_inward, axis = 1)*response_scaling

    # Add baseline activity
    outward_response_time = outward_response_time + baseline_activity
    inward_response_time = inward_response_time + baseline_activity

    if clipping:
        # Clip values below zero so that the 'glutamate signal' cannot go below 0
        outward_response_time = np.clip(outward_response_time, a_min = 0, a_max = None)
        inward_response_time = np.clip(inward_response_time, a_min = 0, a_max = None)

    return outward_response_time, inward_response_time

def run_brian2(stimulus_array, spatial_conversion, temporal_conversion, set_parameters, method_brian, dt_brian):

    dt_data = (1/temporal_conversion)*1000*ms # Temporal conversion is from s to pixel, *1000 for ms
    run_time = stimulus_array.shape[0]*dt_data # In ms

    start_scope()
    # Morphology
    length_dendrite = 150
    number_compartments = int(np.round(length_dendrite*spatial_conversion))
    length_compartments = (length_dendrite/number_compartments)*np.ones(number_compartments)
    d1 = 0.4*np.ones(int(np.round(10*spatial_conversion))) # First 10 micrometer have a bigger diameter
    d2 = 0.2*np.ones(int(number_compartments - np.round(10*spatial_conversion))+1)
    diameter_compartments = np.concatenate((d1,d2))
    morpho = Soma(diameter=7*um)
    morpho.dendrite = Section(diameter = diameter_compartments*um, length = length_compartments*um, n = number_compartments)

    # Equations and parameters
    Ri = 150*ohm*cm
    Cm = 1*uF/cm**2
    Rm = 21700*ohm*cm**2

    EL = -54.387*mV
    ECa = 120*mV
    gCa_ = 0.013 # msiemens/mmeter**2

    Ca0_ = 50*nM # Resting calcium concentration
    tau_ = 5*ms # Time constant for decay of intracellular calcium
    gamma_ = 20*molar/ncoulomb

    ca_distal = gCa_*np.ones(int(np.round(50*spatial_conversion))) # Last 50 um have calcium channels
    ca_proximal = np.zeros(number_compartments+1-int(np.round(50*spatial_conversion))) # No calcium channels
    calcium = np.concatenate((ca_proximal, ca_distal))*msiemens/mmeter**2

    eqs = '''
    Im = (EL - v)/Rm + gCa* m**3 * h * (ECa -v) + stimulus(t, i): amp/meter**2

    dCa/dt = -gamma*ICa*area - (Ca - Ca0)/tau : mM
    ICa =  -gCa * m**3 * h * (ECa-v) : amp/meter**2
    dm/dt = alpham * (1-m) - betam * m : 1
    dh/dt = alphah * (1-h) - betah * h : 1
    alpham = (18570.0*(-0.09/mV)*(v - 67.0*mV)/(exp((-0.09/mV)*(v-67.0*mV)) -1))/1000 /ms : Hz
    betam = (10.0*exp((v-32.0*mV)/(-25.0*mV)))/1000 /ms : Hz
    alphah = (0.36* exp((v+16.0*mV)/(-9.0*mV)))/1000 /ms : Hz
    betah = (0.876*(-0.05/mV)*(v+16.0*mV)/(exp((-0.05/mV)*(v+16.0*mV))-1))/1000 /ms : Hz

    tau : second
    gamma: mM/coulomb
    Ca0 : mM
    gCa : siemens/meter**2
    IStimulus = stimulus(t, i) : amp/meter**2

    '''

    if set_parameters == True:
        neuron = SpatialNeuron(morphology = morpho, model = eqs, Cm = Cm, Ri = Ri, method = method_brian, dt = dt_brian*ms)
    else:
        neuron = SpatialNeuron(morphology = morpho, model = eqs, Cm = Cm, Ri = Ri)

    neuron.gCa = calcium
    neuron.v = EL
    neuron.gamma = gamma_
    neuron.tau = tau_
    neuron.Ca0 = Ca0_
    neuron.Ca = Ca0_
    neuron.h = neuron.alphah/(neuron.alphah + neuron.betah)
    neuron.m = neuron.alpham/(neuron.alpham + neuron.betam)

    area_cell = np.expand_dims(neuron.area, axis = 0)
    stimulus_array_per_Area = stimulus_array*pA/area_cell
    stimulus = TimedArray(stimulus_array_per_Area, dt = dt_data)
    M = StateMonitor(neuron, variables = True, record = True)
    run(run_time)

    return M

def response_extraction(state_monitor, motion_duration, spatial_conversion):
    """
    Extracts the response for outward and inward motion from the state_monitor

    Parameters
    ----------
    state_monitor : brain2 object
        Contains the recorded variables
    motion_duration : float
        Duration of motion stimulus in ms
    spatial_conversion : float
        Ratio to convert from um to pixels
    """

    # Stimulation order:
    # 100 ms of no input
    # 100 ms of background input
    # Outward motion
    # 100 ms of background input
    # Inward motion

    nb_distal_compartments = int(np.round(50*spatial_conversion)) # Last third of the dendrite (50 um)

    index_before_motion = np.where(state_monitor.t <= 200*ms)[0]
    index_outward_motion = np.where((state_monitor.t > 200*ms) & (state_monitor.t <= (200+motion_duration)*ms))[0]
    index_between_motion = np.where((state_monitor.t > (200+motion_duration)*ms) & (state_monitor.t <= (300+motion_duration)*ms))[0]
    index_inward_motion = np.where(state_monitor.t > (300+motion_duration)*ms)[0]

    voltage_outward = state_monitor.v[:,index_outward_motion]/mV
    voltage_inward = state_monitor.v[:,index_inward_motion]/mV
    background_voltage_out = state_monitor.v[:,index_before_motion[-1]]/mV
    background_voltage_in = state_monitor.v[:,index_between_motion[-1]]/mV
    max_voltage_outward = np.amax(voltage_outward, axis = 1) - background_voltage_out
    max_voltage_inward = np.amax(voltage_inward, axis = 1) - background_voltage_in
    DSI_voltage_compartments = (max_voltage_outward - max_voltage_inward)/(max_voltage_outward + max_voltage_inward)
    mean_DSI_voltage = np.mean(DSI_voltage_compartments[-nb_distal_compartments:])

    calcium_outward = state_monitor.Ca[-nb_distal_compartments:,index_outward_motion]/nM
    calcium_inward = state_monitor.Ca[-nb_distal_compartments:,index_inward_motion]/nM
    background_calcium_out = state_monitor.Ca[-nb_distal_compartments:,index_before_motion[-1]]/nM
    background_calcium_in = state_monitor.Ca[-nb_distal_compartments:,index_between_motion[-1]]/nM
    max_calcium_outward = np.amax(calcium_outward, axis = 1) - background_calcium_out
    max_calcium_inward = np.amax(calcium_inward, axis = 1) - background_calcium_in
    DSI_calcium_compartments = (max_calcium_outward - max_calcium_inward)/(max_calcium_outward + max_calcium_inward)
    mean_DSI_calcium = np.mean(DSI_calcium_compartments)

    calciumCurrent_outward = state_monitor.ICa[-nb_distal_compartments:,index_outward_motion]
    calciumCurrent_inward = state_monitor.ICa[-nb_distal_compartments:,index_inward_motion]

    return voltage_outward, voltage_inward, DSI_voltage_compartments, 
            mean_DSI_voltage, calcium_outward, calcium_inward, 
            DSI_calcium_compartments, mean_DSI_calcium, 
            calciumCurrent_outward, calciumCurrent_inward

def run_model(kernels, synapse_locations, extension_right, extension_left,
              outward_stimulus, inward_stimulus,
              spatial_conversion, temporal_conversion,
              baseline_activity = 0, response_scaling = 1, clipping = True,
              set_parameters = False, method_brian = None, dt_brian = None):
    """
    Calculates the stimulus array and runs the Brian2 simulation with it

    Parameters
    ----------
    kernels : list
        List of arrays, each array corresponding to the kernel of one BC
    synapse_locations : list
        List of arrays, each array contains the synapse locations of a BC in um
    extension_right : int/float
        How far the stimulus moves beyond the end of the dendrite in um
    extension_left : int/float
        How far the stimulus moves along the other side of the SAC in um
    outward_stimulus : numpy.ndarray
        Outward stimulus picture
    inward_stimulus : numpy.ndarray
        Inward stimulus picture
    spatial_conversion : float/int
        Contains the conversion from um to pixel
    temporal_conversion : float/int
        Contains the conversion from seconds to pixel
    baseline_activity : float/int
        Spontanous input without stimlus, gets added to result of convolution
    response_scaling : float/int
        Scaling to go from filter activity to ampere
    clipping : boolean
        Whether or not to clip the signal so that input cannot go below 0
    set_parameters : boolean
        Whether or not one provides own method and dt for brian2
    method_brian : str
        Method used for solving differential equations
    dt_brian : float
        Time step used for Brian2 in ms

    Returns
    -------
    state_monitor : brian2 object
        Contains all the recorded values from the simulation

    total_outward_input : numpy.ndarray
        Array of total BC input during outward motion

    total_inward_input : numpy.ndarray
        Array of total BC input during inward motion

    motion_duration_ms : int/float
        Duration of outward/inward motion in ms
    """

    assert len(kernels) == len(synapse_locations)
    nb_kernels = len(kernels) # Number of BCs in model
    total_distance = 150 + extension_right + extension_left # Dendrite is 150; total distance covered by stimulus in um
    total_distance_px = int(np.round(total_distance*spatial_conversion)) # Total distance stimulus covers in pixel
    nb_compartments = int(np.round(150*spatial_conversion)) + 1 # + 1 for soma compartment, size of input array to Brian2
    assert np.isclose(total_distance_px, outward_stimulus.shape[1], atol=1)
    total_baseline_input = np.zeros(nb_compartments)
    total_outward_input = np.zeros((kernels[0].shape[0]+outward_stimulus.shape[0]-1, nb_compartments))
    total_inward_input = np.zeros((kernels[0].shape[0]+inward_stimulus.shape[0]-1, nb_compartments))
    motion_duration_ms = total_inward_input.shape[0]*(1/temporal_conversion)*1000 # in ms

    for kernel_index in range(nb_kernels):

        current_kernel = kernels[kernel_index]
        current_synapse_locations = synapse_locations[kernel_index]

        total_baseline_input += get_baseline_activity(current_synapse_locations, spatial_conversion, baseline_activity)
        outward_response, inward_response = perform_convolution(outward_stimulus, inward_stimulus, current_kernel, baseline_activity, response_scaling, clipping)
        outward_response_array, inward_response_array = create_BC_input_array(outward_response, inward_response, current_synapse_locations, spatial_conversion)
        total_outward_input += outward_response_array
        total_inward_input += inward_response_array

    pre_stimulus_time_period = int(np.round(0.1*temporal_conversion)) # 0.1 s = 100 ms
    no_input = np.repeat(np.zeros((1, nb_compartments)), pre_stimulus_time_period, axis = 0) # 100 ms of no input
    total_baseline_input = np.repeat(total_baseline_input.reshape((1, nb_compartments)), pre_stimulus_time_period, axis = 0) # 100 ms of background input

    # Order: No input, background input, outward motion, background intput, inward motion
    stimuli_combined = np.concatenate((no_input, total_baseline_input, total_outward_input, total_baseline_input, total_inward_input))

    state_monitor = run_brian2(stimuli_combined, spatial_conversion, temporal_conversion, set_parameters, method_brian, dt_brian)

    return state_monitor, total_outward_input, total_inward_input, motion_duration_ms