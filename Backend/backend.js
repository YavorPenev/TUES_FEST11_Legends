const express = require("express");
const mysql = require("mysql2");
require("dotenv").config();
const cors = require("cors");
//const nodemailer = require('nodemailer');
//const crypto = require('crypto');
//const session = require('express-session');
//const path = require('path');

const app = express();

app.use(express.json()); //Middleware за парсване на JSON

app.use(express.urlencoded({ extended: true }));
//app.use(session({ secret: 'securekey', resave: false, saveUninitialized: true }));

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
/*const transporter = nodemailer.createTransport({
    service: 'gmail',
    host: 'smtp.gmail.com',
    port: 587,
    secure: false,
    auth: {
        user: 'deyan662@gmail.com',
        pass: 'sxwh bnvf djli ivtj'
    },
    tls: {
        rejectUnauthorized: false
    }
});

const verificationLinks = {};

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/send-email', (req, res) => {
    const { email, action } = req.body;
    let subject = 'Вашето заявено имейл съдържание';
    
    if (action === 'text') {
        htmlContent = '<p>Здравейте от нашия сайт!</p>';
    } else if (action === 'link') {
        const token = crypto.randomBytes(20).toString('hex');
        verificationLinks[token] = email;
        htmlContent = `<p>Щракнете <a href='http://localhost:8000/homepage/${token}'>тук</a> за достъп до сайта.</p>`;
    } else if (action === 'text-link-image') {
        const token = crypto.randomBytes(20).toString('hex');
        verificationLinks[token] = email;
        htmlContent = `<p>Добре дошли!</p>
                       <p>Щракнете <a href='http://localhost:8000/homepage/${token}'>тук</a> за достъп до сайта.</p>
                       <p>Линк към <a href='https://www.fortworthtexas.gov/files/assets/public/v/1/hr/images/verification-banner.jpg?dimension=pageimagefullwidth&w=1140'>снимката.</a></p>
                       <img src='https://www.fortworthtexas.gov/files/assets/public/v/1/hr/images/verification-banner.jpg?dimension=pageimagefullwidth&w=1140'>`;
    }
    
    const mailOptions = {
        from: 'deyan662@gmail.com',
        to: email,
        subject: 'Имейл за верификация',
        html: htmlContent
    };
    
    transporter.sendMail(mailOptions, (err, info) => {
        if (err) {
            console.error('Грешка при изпращането на имейл:', err);
            res.status(500).send('Грешка при изпращането на имейл. Моля, проверете конзолата за подробности.');
        } else {
            console.log('Имейл изпратен успешно:', info.response);
            res.send('<h1>Имейлът беше изпратен успешно. Моля, проверете пощата си!</h1>');
        }
    });    
});

app.get('/homepage/:token', (req, res) => { //Proverete go dali e homepage(to ne e ama go opravete) + ezika ako trqbva shte go napravq na angliisji prosto me murzeshe
    const { token } = req.params;
    if (verificationLinks[token]) {
        req.session.verified = true;
        res.send('<h1>Добре дошли на главната страница!</h1>');
    } else {
        res.status(403).send('Достъпът е отказан! Моля, използвайте линка за верификация.');
    }
});*/

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);// port na backend
});