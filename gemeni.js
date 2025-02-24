// To generate content, use this import path for GoogleGenerativeAI.
// Note that this is a different import path than what you use for the File API.
// import { GoogleGenerativeAI } from "@google/generative-ai";
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { GoogleAIFileManager, FileState } = require("@google/generative-ai/server");
const dotenv = require('dotenv');
const fs = require('fs');
const fetch = require('node-fetch');
const path = require('path');

dotenv.config();


    // Initialize GoogleGenerativeAI with your GEMINI_API_KEY.
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Initialize GoogleAIFileManager with your GEMINI_API_KEY.
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);

async function uploadVideoFile(videoPath) {
    // Check if the file exists
    if (!fs.existsSync(videoPath)) {
        throw new Error(`File not found: ${videoPath}`);
    }

    // Upload the file and specify a display name.
    const uploadResponse = await fileManager.uploadFile(videoPath, {
        mimeType: "video/mp4",
        displayName: path.basename(videoPath),
    });

    // View the response.
    console.log(`Uploaded file ${uploadResponse.file.displayName} as: ${uploadResponse.file.uri}`);

    return uploadResponse.file;
}

async function waitForFileToBeActive(fileName) {
    let file = await fileManager.getFile(fileName);
    while (file.state === FileState.PROCESSING) {
        process.stdout.write(".");
        // Sleep for 10 seconds
        await new Promise((resolve) => setTimeout(resolve, 10_000));
        // Fetch the file from the API again
        file = await fileManager.getFile(fileName);
    }

    if (file.state === FileState.FAILED) {
        throw new Error("Video processing failed.");
    }

    // When file.state is ACTIVE, the file is ready to be used for inference.
    console.log(`\nFile ${file.displayName} is ready for inference as ${file.uri}`);
    return file.uri;
}

async function analyzeVideo(videoPath) {
    console.log("Analyzing video: " + videoPath);

    // Upload the video file using the File API
    const uploadedFile = await uploadVideoFile(videoPath);

    // Wait for the file to be in ACTIVE state
    const fileUri = await waitForFileToBeActive(uploadedFile.name);

    // Choose a Gemini model.
    const model = genAI.getGenerativeModel({
        model: "gemini-1.5-pro",
    });

    // Generate content using text and the URI reference for the uploaded file.
    const result = await model.generateContent([
        {
            fileData: {
                mimeType: 'video/mp4',
                fileUri: fileUri
            }
        },
        { text: `
            Describe the video in detail. You are a trained security agent performing a demo.
                            - Identify potential hiding spots and vantage points, access routes and vulnerabilities.
                            - Look for the most likely hiding 
                            -Property Overview
                            - Object Classification
                                - People 
                                - Vehicles
                                - Gatherings / groups
                                - Open gates / fences / unsecured access to area
                                - Unattended objects
                            - Motion & Anomaly Detection:
                                - Crowd density estimation
                                - Unexpected activity detection
                            
                            Add risk scores to each item you analyze in the video on a scale of 0-100 based on the likelihood of it being a security risk where 0 is no risk and 100 is a high risk.
` },
    ]);

    // Handle the response of generated text
    console.log(result.response.text());
}

analyzeVideo("./test2.mp4");