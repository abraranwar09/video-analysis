const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

const app = express();
const port = 3000;

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Configure OpenAI client
const openai = new OpenAI({
    // apiKey: process.env.OPENAI_API_KEY
    baseURL: 'http://localhost:11434/v1',
    apiKey: 'ollama'
});   


// Endpoint to receive video files
app.post('/video-analysis', upload.single('video'), async (req, res) => {
    console.log('Received video file:', req.file);
    const videoPath = req.file.path;
    const nthFrame = 10; // Define nthFrame to extract every 10th frame, adjust as needed
    try {
        const analysisResults = await analyzeVideo(videoPath, nthFrame);
        fs.writeFileSync('analysis.json', JSON.stringify(analysisResults, null, 2));
        console.log('Analysis results saved to analysis.json');
        res.send('Video analysis complete. Results saved to analysis.json');
    } catch (error) {
        console.error('Error during video analysis:', error);
        res.status(500).send('An error occurred during video analysis.');
    }
});

// Function to analyze video
async function analyzeVideo(videoPath, nthFrame) {
    console.log(`Starting video analysis for: ${videoPath}`);
    const frames = await extractFrames(videoPath, nthFrame);
    const results = [];

    for (const frame of frames) {
        console.log(`Processing frame: ${frame}`);
        const analysis = await analyzeFrameWithLlama(frame);
        results.push(analysis);
        console.log(`Analysis results: ${analysis}`);
    }

    console.log('Video analysis complete:', results);
    return results;
}

// Function to extract frames from video using ffmpeg
async function extractFrames(videoPath, nthFrame) {
    console.log(`Extracting frames from video: ${videoPath}, every ${nthFrame}th frame`);
    const outputDir = path.join(__dirname, 'frames');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
        console.log(`Created directory for frames: ${outputDir}`);
    }

    return new Promise((resolve, reject) => {
        const command = `ffmpeg -i ${videoPath} -vf "select=not(mod(n\\,${nthFrame}))" -vsync vfr ${outputDir}/frame_%04d.png`;
        console.log(`Running command: ${command}`);

        require('child_process').exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error extracting frames: ${stderr}`);
                return reject(error);
            }
            console.log(`Frames extracted to ${outputDir}`);
            fs.readdir(outputDir, (err, files) => {
                if (err) {
                    console.error('Error reading frames directory:', err);
                    return reject(err);
                }
                const framePaths = files.map(file => path.join(outputDir, file));
                console.log(`Extracted frames: ${framePaths}`);
                resolve(framePaths);
            });
        });
    });
}

// Function to encode an image to Base64
function encodeImageToBase64(imagePath) {
    console.log(`Encoding image to Base64: ${imagePath}`);
    const imageBuffer = fs.readFileSync(imagePath);
    return imageBuffer.toString('base64');
}

// Function to analyze a frame with LLaMA-3.2 Vision
async function analyzeFrameWithLlama(framePath) {
    console.log(`Analyzing frame: ${framePath}`);
    const base64Image = encodeImageToBase64(framePath);

    try {
        const response = await openai.chat.completions.create({
            model: "llama3.2-vision",
            messages: [
                {
                    role: "user",
                    content: [
                        { type: "text", text: `Describe the image in detail. You are a trained security agent performing a demo.
                            - Identify potential hiding spots and vantage points, access routes and vulnerabilities.
                            - Look for the most likely hiding 
                            - Object Classification
                            - People 
                            - Vehicles
                            - Gatherings / groups
                            - Open gates / fences / unsecured access to area
                            - Unattended objects
                            - Motion & Anomaly Detection:
                            - Crowd density estimation
                            - Unexpected activity detection
                            ` },
                        {
                            type: "image_url",
                            image_url: {
                                url: `data:image/png;base64,${base64Image}`,
                                detail: "auto"
                            }
                        }
                    ]
                }
            ],
            max_tokens: 1000,
            temperature: 0.5,
        });

        // Log the entire response for debugging
        console.log('API Response:', response);

        // Check if choices exist in the response
        if (response.choices) {
            console.log('Response choices:', response.choices[0].message.content);
            // return response.choices[0].message;
        } else {
            throw new Error('Unexpected API response structure');
        }
    } catch (error) {
        console.error('Error analyzing frame:', error);
        throw error;
    }
}

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
