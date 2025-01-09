%% Acoustic Noise Cancellation (LMS)
% This example shows how to use the least mean square (LMS) algorithm to
% subtract noise from an input signal. The example uses a preconfigured
% Simulink(R) model. The LMS Filter block in the |dspanc| model models an
% adaptive filter that uses the reference signal at its *Input* port and
% the desired signal at the *Desired* port to automatically match the
% filter response. The LMS Filter block subtracts the filtered noise from
% the original signal. As the filter converges, the resultant error signal
% contains only the original signal.

% Copyright 2006-2023 The MathWorks, Inc.

%% Acoustic Noise Canceler Model
% In the |dspanc| model used in this example, the signal output at the
% upper port of the |Acoustic Environment| subsystem is white noise. The
% signal output at the lower port is composed of colored noise and a signal
% from a WAV file. This model uses an adaptive filter to remove the noise
% from the signal output at the lower port. When you run the simulation,
% you hear both noise and a person playing the drums. Over time, the
% adaptive filter in the model filters out the noise so you hear only the
% drums.
%
% Open the |dspanc| model.

%% Utilize Your Audio Device
% Run the model and use your audio device to listen to the audio signal in
% real time (while running the simulation). The stop time is set to
% infinity, which allows you to interact with the model while it is
% running. For example, you can change the filter or alternate between slow
% adaptation and fast adaptation, and get a sense of the real-time audio
% processing behavior under these conditions.
%%
open_system('dspanc');
set_param('dspanc','StopTime','100');
sim('dspanc');
%%
bdclose dspanc;

%% Color Codes of the Blocks
% When you run the model, the color codes of the blocks in the model
% indicate how fast a block executes. In this model, the color red
% indicates the fastest discrete sample time (for example, the 8 kHz audio
% signal processing block in the |Acoustic Environment| subsystem) and
% the color green indicates the second fastest discrete sample time. You can see that
% the color changes from red to green in the |Array Plot| block after
% downsampling by 32. For more information on displaying sample time
% colors, see <docid:simulink_ug#bry66ow View Sample Time Information>.

%% Array Plot Scope
% Double-click the |Array Plot| block to open the scope window and display
% the behavior of the coefficients of the adaptive filter. The window
% displays multiple samples of data at one time. These samples represent
% the values of the filter coefficients of a normalized LMS adaptive
% filter. The Array Plot window has toolbar buttons that enable you to zoom
% in on the displayed data, freeze the scope display, and save the scope
% position.

%% Acoustic Environment Subsystem
% You can see the details of the |Acoustic Environment| subsystem by
% double-clicking the subsystem in the model. Gaussian noise is used to
% create the signal sent to the *Exterior Mic* output port. If the input to
% the *Filter* port changes from 0 to 1, the Digital Filter block changes
% from a lowpass filter to a bandpass filter. The filtered noise output
% from the Digital Filter block is added to the signal coming from a WAV
% file to produce the signal sent to the *Pilot's Mic* output port.

%% Available Example Versions
% For a fixed-point version of this model, open the |dspanc_fixpt| model
% included with this example.
