require("dotenv").config();
const express = require("express");
const mysql = require("mysql2");
const bcrypt = require("bcrypt");
const path = require("path");
const axios = require("axios");
const nodemailer = require("nodemailer");
const { OpenAI } = require("openai");
const session = require("express-session");
const bodyParser = require("body-parser");
const cors = require('cors');
//const nodemon = require('nodemon');
const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY;
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const app = express();

app.use(express.json()); //Middleware за парсване на JSON

app.use(express.urlencoded({ extended: true }));


////////////////////////////////////////////////////

const corsOptions = {
origin:"http://localhost:5173",// wruzka s frontend
};

app.use(cors(corsOptions));

////////////////////////////////////////////////////

const db = mysql.createConnection({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    port: process.env.DB_PORT,
});

db.connect((err) => {
    if (err) {
        console.error("Error connecting to the database:", err);
        process.exit(1); // Exit the application if the database connection fails
    } else {
        console.log("Connected to the database");
    }
});

/////////////////////////////////////////////////////

app.post("/add", (req, res) => {
    const { title, body } = req.body;// превеъща в json формат

    if (!title || !body) {
        return res.status(400).json({ error: "Title and body are required" });
    }

    const sql = "INSERT INTO notes (title, body) VALUES (?, ?)";
    db.query(sql, [title, body], (err, result) => {
        if (err) {
            console.error("Error inserting note:", err);
            return res.status(500).json({ error: "Failed to insert note" });
        }
        res.status(201).json({ message: "Note added successfully", id: result.insertId });
    });
});

////////////////////////////////////////////////////

app.get("/notes", (req, res) => {
    const sql = "SELECT * FROM notes";
    db.query(sql, (err, results) => {
        if (err) {
            console.error("Error fetching notes:", err);
            return res.status(500).json({ error: "Failed to fetch notes" });
        }
        res.status(200).json(results);
    });
});

/////////////////////////////////////////////////////

app.get("/api", (req, res) => {
res.json( {fruit:["apple", " bannana", "oranges"]}); //testwane na wruzkata
});

////////////////////////////////////////////////////

app.delete("/delete", (req, res) => {
    const { title } = req.body;

    if (!title) {
        return res.status(400).json({ error: "Title is required" });
    }

    const sql = "DELETE FROM notes WHERE title = ?";
    db.query(sql, [title], (err, result) => {
        if (err) {
            console.error("Error deleting note:", err);
            return res.status(500).json({ error: "Failed to delete note" });
        }
        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Note not found" });
        }
        res.status(200).json({ message: "Note deleted successfully" });
    });
});

/////////////////////////////////////////////////////////////

app.put("/edit", (req, res) => {
    const { oldTitle, newTitle, newBody } = req.body;

    if (!oldTitle || !newTitle || !newBody) {
        return res.status(400).json({ error: "Old title, new title, and new body are required" });
    }

    const sql = "UPDATE notes SET title = ?, body = ? WHERE title = ?";
    db.query(sql, [newTitle, newBody, oldTitle], (err, result) => {
        if (err) {
            console.error("Error updating note:", err);
            return res.status(500).json({ error: "Failed to update note" });
        }
        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Note not found" });
        }
        res.status(200).json({ message: "Note updated successfully" });
    });
});

//////////////////////////////////////////////////////////////

async function fetchAllStockSymbols(){
    const exchanges = ["US", "LSE", "HKEX", "BSE", "SSE", "TSE", "KOSDAQ"];

let allsymbols = [];

for(const exchange of exchanges){
    try{
        const response = await axios.get(`https://finnhub.io/api/v1/stock/symbol?exchange=${exchange}&token=${FINNHUB_API_KEY}`);
        const symbols = response.data
        allsymbols = [...allsymbols, ...symbols];
        console.log(`Fetched ${symbols.length} symbols from exchange: ${exchange}`);
    }
    catch(error){
        console.error(`Error fetching stock symbols for exchange ${exchange}:`, error.message);
    }
}
console.log('Total symbols fetched:', allSymbols.length);
return allSymbols;
}

async function getStockData(symbol) {
    try {
        const response = await axios.get(`https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${FINNHUB_API_KEY}`);
        const data = response.data;

        if (!data) throw new Error("No data found for symbol");

        return {
            symbol: symbol,
            currentPrice: data.c,
            high: data.h,
            low: data.l,
            open: data.o,
            close: data.pc,
        };
    } catch (err) {
        console.error("Error fetching stock data:", err.message);
        return null;
    }
}
async function getInvestmentAdvice(userProfile) {
    const stockSymbols = await fetchAllStockSymbols(); // Fetch all stock symbols

    const stockDataPromises = stockSymbols.slice(0, 5).map(symbol => getStockData(symbol.symbol)); // Only first 5 symbols

    try {
        const stockData = await Promise.all(stockDataPromises);
        const filteredStockData = stockData.filter(data => data !== null);

        const prompt = `
        Based on the following stock data for the given stocks:

        ${filteredStockData.map(data => `
        Stock: ${data.symbol}
        Current Price: ${data.currentPrice}
        High: ${data.high}
        Low: ${data.low}
        Open: ${data.open}
        Close: ${data.close}`).join("\n")}

        and the user's financial profile:
        Income: ${userProfile.income}
        Expenses: ${userProfile.expenses}
        Goals: ${userProfile.goals.join(", ")}

        Provide a detailed recommendation on which stock(s) would be the best investment for the user.
        Consider factors such as stock performance, risk, and user preferences.
        `;

        const response = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [{ role: "user", content: prompt }],
        });

        return response.choices[0].message.content;
    } catch (err) {
        console.error("Error getting investment advice:", err.message);
        return "An error occurred while processing the investment advice.";
    }
}


/////////////////////////////////////////////////////////////
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);// port na backend
});