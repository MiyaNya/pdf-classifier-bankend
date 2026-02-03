// backend/server.js
require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');

let pdf; // จะโหลดด้วย dynamic import
let pdfjsLib; // สำหรับ extract text ทีละหน้า

const app = express();
const port = process.env.PORT || 5000;

// ตั้งค่า Middleware
app.use(cors());
app.use(express.json());

// ตั้งค่าการอัปโหลดไฟล์ด้วย Multer (เก็บใน RAM ชั่วคราว) - รองรับหลายไฟล์
const storage = multer.memoryStorage();
const upload = multer({ 
    storage: storage,
    limits: { fileSize: 50 * 1024 * 1024 } // จำกัด 50MB ต่อไฟล์
});

// ตั้งค่า OpenRouter Client (ใช้ Grok ผ่าน OpenRouter)
const openrouter = new OpenAI({
    baseURL: 'https://openrouter.ai/api/v1',
    apiKey: process.env.XAI_API_KEY,
    defaultHeaders: {
        'HTTP-Referer': 'http://localhost:5173',
        'X-Title': 'PDF Classifier'
    }
});

// ฟังก์ชันสำหรับ extract text จาก PDF ทีละหน้า
async function extractTextByPage(buffer) {
    const uint8Array = new Uint8Array(buffer);
    const loadingTask = pdfjsLib.getDocument({ data: uint8Array });
    const pdfDocument = await loadingTask.promise;
    
    const pageTexts = [];
    
    for (let i = 1; i <= pdfDocument.numPages; i++) {
        const page = await pdfDocument.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items.map(item => item.str).join(' ');
        pageTexts.push({
            pageNumber: i,
            text: pageText
        });
    }
    
    return pageTexts;
}

// ฟังก์ชันค้นหาหน้าบทคัดย่อ
function findAbstractPages(pageTexts) {
    const abstractKeywords = ['บทคัดย่อ', 'ABSTRACT', 'Abstract', 'บท คัดย่อ'];
    const abstractPages = [];
    
    for (let i = 0; i < pageTexts.length; i++) {
        const text = pageTexts[i].text;
        // ตรวจสอบว่ามีคำว่า บทคัดย่อ หรือ ABSTRACT ในหน้านี้หรือไม่
        const hasAbstract = abstractKeywords.some(keyword => 
            text.toLowerCase().includes(keyword.toLowerCase())
        );
        
        if (hasAbstract) {
            // เก็บหน้านี้และหน้าถัดไป 2-3 หน้า (เผื่อบทคัดย่อยาว)
            abstractPages.push(i);
            // เพิ่มหน้าถัดไป 2 หน้า
            if (i + 1 < pageTexts.length) abstractPages.push(i + 1);
            if (i + 2 < pageTexts.length) abstractPages.push(i + 2);
            break; // หยุดเมื่อเจอหน้าแรกที่มีบทคัดย่อ
        }
    }
    
    return abstractPages;
}

// ฟังก์ชันแปลง model name ที่รับจาก frontend ไปเป็น model ID ของ OpenRouter
function getModelConfig(modelName) {
    const modelConfigs = {
        'Nvidia-AI': {
            id: 'nvidia/nemotron-3-nano-30b-a3b:free',
            name: 'Nvidia Nemotron 3 Nano 30B A3B',
            temperature: 0.1
        },
    };
    
    return modelConfigs[modelName] || modelConfigs['Nvidia-AI'];
}

// --- API Endpoint สำหรับการจำแนกเอกสาร (รองรับหลายไฟล์) ---
app.post('/api/classify', upload.array('pdfFiles', 20), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'กรุณาอัปโหลดไฟล์ PDF อย่างน้อย 1 ไฟล์' });
        }

        // รับ model ที่เลือกจาก frontend
        const selectedModel = req.body.model || 'deepseek';
        const modelConfig = getModelConfig(selectedModel);
        
        console.log(`Processing ${req.files.length} file(s) with ${modelConfig.name}...`);
        
        const results = [];

        // ประมวลผลทีละไฟล์
        for (let fileIndex = 0; fileIndex < req.files.length; fileIndex++) {
            const file = req.files[fileIndex];
            
            // แก้ไข encoding ของชื่อไฟล์ภาษาไทย
            const filename = Buffer.from(file.originalname, 'latin1').toString('utf8');
            
            console.log(`\n[${fileIndex + 1}/${req.files.length}] Processing: ${filename}`);

            try {
                // 1. ดึงข้อความจาก PDF ทีละหน้า
                const dataBuffer = file.buffer;
                console.log("  - Extracting text from PDF pages...");
                const pageTexts = await extractTextByPage(dataBuffer);
                
                console.log(`  - Total pages: ${pageTexts.length}`);
                
                // 2. ค้นหาหน้าบทคัดย่อ
                const abstractPageIndices = findAbstractPages(pageTexts);
                
                let extractedText = '';
                
                if (abstractPageIndices.length > 0) {
                    console.log(`  - Found abstract pages: ${abstractPageIndices.map(i => pageTexts[i].pageNumber).join(', ')}`);
                    // รวม text จากหน้าบทคัดย่อ
                    extractedText = abstractPageIndices
                        .map(i => pageTexts[i].text)
                        .join('\n\n');
                } else {
                    console.log("  - Abstract pages not found, using first 3 pages...");
                    // ถ้าไม่เจอบทคัดย่อ ให้ใช้ 3 หน้าแรก
                    extractedText = pageTexts.slice(0, 3)
                        .map(p => p.text)
                        .join('\n\n');
                }

                // จำกัดความยาวไม่เกิน 6000 ตัวอักษร
                const truncatedText = extractedText.substring(0, 6000);

                console.log("  - Sending to xAI Grok...");

                // 3. ส่งไปยัง xAI Grok API
                const categories = ['Web-application', 'Mobile-application', 'Hardware/IoT & Network', 'Digital Image Processing', 'Other'];
                
                const completion = await openrouter.chat.completions.create({
                    messages: [
                        {
                            role: "system",
                            content: `You are an expert document classifier for university senior projects. Your task is to analyze the provided abstract text from a Thai university thesis and categorize it into ONLY ONE of the following categories: ${categories.join(', ')}. 

Guidelines:
- Web-application: Projects involving web applications, websites, web services, APIs, web development
- Mobile-application: Projects involving iOS, Android, mobile applications, mobile development
- Hardware/IoT & Network: Projects involving IoT, embedded systems, robotics, Arduino, sensors, networking, network infrastructure
- Digital Image Processing: Projects involving computer vision, image analysis, image processing, OCR, face recognition, pattern recognition
- Other: Projects that don't fit the above categories

Output ONLY the category name. Do not add any explanation.`
                        },
                        {
                            role: "user",
                            content: `Here is the document abstract:\n\n${truncatedText}`
                        }
                    ],
                    model: modelConfig.id,
                    temperature: modelConfig.temperature
                });

                const category = completion.choices[0].message.content.trim();
                
                console.log(`  - Result: ${category}`);

                // เก็บผลลัพธ์
                results.push({
                    filename: filename, 
                    category: category,
                    pagesProcessed: abstractPageIndices.length > 0 
                        ? abstractPageIndices.map(i => pageTexts[i].pageNumber) 
                        : [1, 2, 3],
                    success: true
                });

            } catch (fileError) {
                console.error(`  - Error processing ${filename}:`, fileError.message);
                results.push({
                    filename: filename, 
                    category: 'Error',
                    error: fileError.message,
                    success: false
                });
            }
        }

        console.log(`\nCompleted processing ${results.length} file(s)`);

        // 4. ส่งผลลัพธ์ทั้งหมดกลับไปที่ Frontend
        res.json({ 
            total: results.length,
            results: results
        });

    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'เกิดข้อผิดพลาดในการประมวลผลไฟล์' });
    }
});

// เริ่มต้น server หลังจากโหลด pdf-parse และ pdfjs
(async () => {
    pdf = (await import('pdf-parse')).default;
    pdfjsLib = await import('pdfjs-dist/legacy/build/pdf.mjs');
    
    app.listen(port, () => {
        console.log(`Backend server running on port ${port}`);
        console.log(`AI Model: xAI Grok Beta (Free)`);
        console.log(`Text extraction: Using pdfjs-dist for page-by-page extraction`);
    });
})();