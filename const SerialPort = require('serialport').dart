const SerialPort = require('serialport');
const port = new SerialPort('/dev/ttyS0', { baudRate: 9600 });

const expectedStates = [0xA9, 0x25, 0x55, 0x4A, 0x54, 0x52, 0xA5, 0x29, 0x2A]; // Prior states
const receivedStates = [];
let elevation = 16, k = 1;

port.on('data', (data) => {
    receivedStates.push(...data); // Read 0x53, 0x54, 0x55
    console.log('Received:', receivedStates.map(s => `0x${s.toString(16).padStart(2, '0')}`));

    receivedStates.forEach((state, i) => {
        if (expectedStates.includes(state)) {
            console.log(`Match: 0x${state.toString(16).padStart(2, '0')} at index ${i}`);
        }
        let altitude = elevation + k * state;
        console.log(`State: 0x${state.toString(16).padStart(2, '0')}, Altitude: ${altitude}`);
    });
});

port.on('error', (err) => console.error('Error:', err));