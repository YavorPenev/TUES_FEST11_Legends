const express = require("express");
const mysql = require("mysql2");
require("dotenv").config();
const app = express();
const cors = require("cors");

app.use(express.json()); //Middleware за парсване на JSON

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

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);// port na backend
});