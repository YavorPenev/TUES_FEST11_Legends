require("dotenv").config();
const express = require("express");
const mysql = require("mysql2");
const bcrypt = require("bcrypt");
const path = require("path");
const axios = require("axios");
const nodemailer = require("nodemailer");
const crypto = require("crypto");
const { OpenAI } = require("openai");
const session = require("express-session");
const bodyParser = require("body-parser");
const cors = require('cors');
const multer = require("multer");
const fs = require("fs");


const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY;
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Session setup
app.use(session({
    secret: 'securekey',
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: false, // true в продукция с HTTPS
        httpOnly: true,
        sameSite: 'lax',
        maxAge: 1000 * 60 * 60 * 24 // 1 ден
    }
}));

// CORS setup
const corsOptions = {
    origin: "http://localhost:5173",
    credentials: true,
};
app.use(cors(corsOptions));

// Database connection
const db = mysql.createConnection({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    port: process.env.DB_PORT,
});

db.connect((err) => {
    if (err) {
        console.error("Database connection failed:", err);
        process.exit(1);
    } else {
        console.log("Connected to database");
    }
});

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "uploads/"); // Папка за качване на файлове
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + "-" + file.originalname); // Уникално име за файла
    },
});

const upload = multer({ storage });

const linkStorage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "uploads/links/");
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + "-" + file.originalname);
    },
});

const uploadLinkImage = multer({ storage: linkStorage });

// ---------------------- Middleware ----------------------

function isAuthenticated(req, res, next) {
    if (req.session && req.session.user) {
        next();
    } else {
        console.error("User is not authenticated");
        res.status(401).json({ error: "Access denied. Please log in to access this resource." });
    }
}


// --------------------------- Helpers ---------------------------

/*async function fetchAllStockSymbols() {
    const exchanges = ["US", "LSE", "HKEX", "BSE", "SSE", "TSE", "KOSDAQ"];
    let allsymbols = [];

    for (const exchange of exchanges) {
        try {
            const response = await axios.get(`https://finnhub.io/api/v1/stock/symbol?exchange=${exchange}&token=${FINNHUB_API_KEY}`);
            const symbols = response.data;
            allsymbols = [...allsymbols, ...symbols];
        } catch (error) {
            console.error(`Error fetching symbols from ${exchange}:`, error.message);
        }
    }
    return allsymbols;
}

async function getStockData(symbol) {
    try {
        const response = await axios.get(`https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${FINNHUB_API_KEY}`);
        const data = response.data;
        return {
            symbol,
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
}*/

/*async function getInvestmentAdvice(userProfile) {
    const stockSymbols = await fetchAllStockSymbols();
    const stockDataPromises = stockSymbols.slice(0, 5).map(symbol => getStockData(symbol.symbol));

    try {
        const stockData = await Promise.all(stockDataPromises);
        const filteredStockData = stockData.filter(data => data !== null);

        const prompt = `
        Based on the following stock data:

        ${filteredStockData.map(data => `
        Stock: ${data.symbol}
        Current Price: ${data.currentPrice}
        High: ${data.high}
        Low: ${data.low}
        Open: ${data.open}
        Close: ${data.close}`).join("\n")}

        And user profile:
        Income: ${userProfile.income}
        Expenses: ${userProfile.expenses}
        Goals: ${userProfile.goals.join(", ")}

        Suggest which stocks are the best investments and why.
        `;

        const response = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [{ role: "user", content: prompt }],
        });

        return response.choices[0].message.content;
    } catch (err) {
        console.error("AI error:", err.message);

        if (err.status === 401) {
            console.error("Access denied. Please log in to access this resource.");
        } else {
            console.error("Failed to generate advice.");
        }

        return "Failed to generate advice.";
    }
}*/

app.post("/budgetplanner", async (req, res) => {
    const { income, expenses, familySize, goals, savingsGoalAmount, savingsGoalPeriodMonths } = req.body;

    if (!income || !expenses || !familySize || !goals || goals.trim() === "") {
        return res.status(400).json({ error: "Income, expenses, family size, and goals are required." });
    }

    const availableBudget = income - expenses;
    const monthlySavingsNeeded = savingsGoalAmount / savingsGoalPeriodMonths;

    if (monthlySavingsNeeded > availableBudget) {
        return res.status(400).json({
            error: `With the current budget, your financial goal is not achievable. You need $${monthlySavingsNeeded.toFixed(2)} per month, but you only have $${availableBudget.toFixed(2)} available.`,
        });
    }

    const foodPerPersonPerDay = 10;
    const foodBudgetMonthly = foodPerPersonPerDay * familySize * 30;

    if (foodBudgetMonthly > availableBudget) {
        return res.status(400).json({
            error: `With the current budget, you cannot afford basic food expenses for ${familySize} family members.`,
        });
    }

    const remainingBudgetAfterFood = availableBudget - foodBudgetMonthly;

    if (monthlySavingsNeeded > remainingBudgetAfterFood) {
        return res.status(400).json({
            error: `After covering food expenses, you cannot meet your savings goal.`,
        });
    }

    const distributeRemainingBudget = (remainingBudget) => {
        const essentialsPercentage = 0.4;
        const entertainmentPercentage = 0.2;
        const savingsPercentage = 0.3;
        const investmentsPercentage = 0.1;

        return {
            essentials: remainingBudget * essentialsPercentage,
            entertainment: remainingBudget * entertainmentPercentage,
            savings: remainingBudget * savingsPercentage,
            investments: remainingBudget * investmentsPercentage,
        };
    };

    const { essentials, entertainment, savings, investments } = distributeRemainingBudget(remainingBudgetAfterFood);

    const prompt = `
  Act as a professional financial advisor.

    Based on the following user's data:
    - Monthly Income: $${income}
    - Monthly Fixed Expenses: $${expenses}
    - Available Budget after Fixed Expenses: $${availableBudget}
    - Family Members: ${familySize}
    - Food Budget (fixed): $${foodBudgetMonthly}
    - Remaining Budget After Food: $${remainingBudgetAfterFood}
    - Savings Goal: $${savingsGoalAmount} over ${savingsGoalPeriodMonths} months

    Create a clear and detailed financial plan divided into **THREE SECTIONS**:

    ---
    SECTION 1: Budget Overview
    - Briefly explain:
      - Monthly income
      - Monthly fixed expenses
      - Monthly food budget
      - Remaining available budget for other purposes (investments, entertainment, savings, etc.)
      - Monthly savings required to achieve the goal

    ---
    SECTION 2: Daily Budget Plan (for 30 days)
    - For each day (Day 1 to Day 30), list:
      - Food: $X.XX
      - Investments: $X.XX
      - Entertainment: $X.XX
      - Other Essentials: $X.XX
      - Savings: $X.XX
    - Each day must be listed separately (without grouping - this means that you will have to print a total of thirty messages).
    - The daily values must add up perfectly so that the monthly total matches the available budget after fixed expenses.
    - All numbers must be realistic and rounded to two decimals.

    Strict example format:

    Day 1:
      - Food: $X.XX
      - Investments: $X.XX
      - Entertainment: $X.XX
      - Other Essentials: $X.XX
      - Savings: $X.XX

    Day 2:
    ...

    (Continue for all 30 days - this means that you will have to print a total of thirty messages)

    ---
    SECTION 3: Conclusion and Advice
    - If the goal is achievable:
      - Write a short, motivational paragraph encouraging the user to stay disciplined and consistent with their financial plan.
    - If the goal is NOT achievable (meaning remaining budget is insufficient for the savings goal):
      - Clearly state that it is impossible to meet the goal under the current conditions.
      - Suggest practical advice like reducing expenses, increasing income, or adjusting the savings timeline.

    ---
    Important:
    - Be extremely clear, practical, and well-structured.
    - Never skip a day.
    - Strictly follow the three-section structure.
    `;

    try {
        const response = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [{ role: "user", content: prompt }],
            temperature: 0.6,
            max_tokens: 3500,
        });

        const budgetPlan = response.choices[0].message.content;

        res.status(200).json({ plan: budgetPlan });
    } catch (err) {
        console.error("Error generating budget plan:", err);
        res.status(500).json({ error: "Failed to generate budget plan" });
    }
});


app.post("/model-advice", isAuthenticated, async (req, res) => {
    try {
        const { income, expenses, goal, timeframe } = req.body;

        const response = await axios.post('http://localhost:5001/predict', {
            income: parseFloat(income),
            expenses: parseFloat(expenses),
            goal: parseFloat(goal),
            timeframe: parseInt(timeframe)
        });

        if (!response.data.success) {
            return res.status(500).json({
                error: response.data.error || 'prediction_failed'
            });
        }

        const formattedAdvice = response.data.recommendations.map(rec =>
            `Stock: ${rec.name} (${rec.symbol})\n` +
            `Recommended Investment: $${parseFloat(rec.recommended_amount).toFixed(2)}\n` +
            `Predicted Price: $${parseFloat(rec.predicted_price).toFixed(2)}\n` +
            `Predicted Return: ${parseFloat(rec.predicted_return).toFixed(2)}%\n` +
            `Timeframe: ${rec.timeframe} months\n` +
            `------------------------`
        ).join('\n');

        res.status(200).json({
            success: true,
            advice: formattedAdvice,
            rawData: response.data
        });

    } catch (error) {
        console.error("Model error:", error);
        const status = error.response?.status || 500;
        res.status(status).json({
            error: error.response?.data?.error || 'server_error',
            message: error.response?.data?.message || 'Failed to process request'
        });
    }
});


// ---------------------- Routes ----------------------

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "index.html"));
});

app.post("/signup", async (req, res) => {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
        return res.status(400).json({ error: "All fields are required" });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, 10);
        const sql = "INSERT INTO users (username, email, password) VALUES (?, ?, ?)";
        db.query(sql, [username, email, hashedPassword], (err, result) => {
            if (err) {
                if (err.code === "ER_DUP_ENTRY") {
                    return res.status(400).json({ error: "Username or email already exists" });
                }
                return res.status(500).json({ error: "Database error" });
            }

            const token = crypto.randomBytes(20).toString("hex");
            verificationLinks[token] = email;

            const htmlContent = `
            <p>Welcome, ${username}!</p>
            <p>Click <a href="http://localhost:8000/verify/${token}">here</a> to verify your account.</p>
            `;

            const mailOptions = {
                from: process.env.USER_EMAIL,
                to: email,
                subject: 'Account Verification',
                html: htmlContent,
            };

            transporter.sendMail(mailOptions, (err, info) => {
                if (err) {
                    return res.status(500).json({ error: "Failed to send verification email" });
                }
                res.status(201).json({ message: "User created. Check your email for verification." });
            });
        });
    } catch (error) {
        res.status(500).json({ error: "Signup error" });
    }
});

app.post("/login", (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ error: "Username and password required" });
    }

    const sql = "SELECT * FROM users WHERE username = ?";
    db.query(sql, [username], async (err, results) => {
        if (err) return res.status(500).json({ error: "Database error" });

        if (results.length === 0) return res.status(404).json({ error: "User not found" });

        const user = results[0];
        const isMatch = await bcrypt.compare(password, user.password);

        if (!isMatch) {
            return res.status(401).json({ error: "Invalid credentials" });
        }

        req.session.user = {
            id: user.id,
            username: user.username,
            email: user.email,
        };

        res.status(200).json({ message: "Login successful", user: req.session.user });
    });
});

const verificationLinks = {};

const transporter = nodemailer.createTransport({
    service: 'gmail',
    host: 'smtp.gmail.com',
    port: 587,
    secure: false,
    auth: {
        user: process.env.USER_EMAIL,
        pass: process.env.APP_PASS,
    },
    tls: {
        rejectUnauthorized: false,
    },
});

app.get("/verify/:token", (req, res) => {
    const token = req.params.token;
    const email = verificationLinks[token];

    if (email) {
        delete verificationLinks[token];
        res.redirect("http://localhost:5173");
    } else {
        res.status(403).send("Invalid or expired verification link.");
    }
});

app.post("/advice", isAuthenticated, async (req, res) => {
    const { income, expenses, goals } = req.body;

    if (!income || !expenses || !goals) {
        return res.status(400).json({ error: "Income, expenses, and goals required" });
    }

    try {
        const advice = await getInvestmentAdvice({ income, expenses, goals });
        res.status(200).json({ advice });
    } catch (error) {
        res.status(500).json({ error: "Failed to generate advice" });
    }
});

app.post("/invest", isAuthenticated, async (req, res) => {
    const { investments, goals } = req.body;

    if (!Array.isArray(investments) || !Array.isArray(goals)) {
        return res.status(400).json({ error: "Investments and goals required." });
    }

    try {
        const stockData = await Promise.all(
            investments.map(async ({ symbol, amount }) => {
                const quoteResponse = await axios.get(`https://finnhub.io/api/v1/quote`, {
                    params: { symbol, token: FINNHUB_API_KEY },
                });
                const profileResponse = await axios.get(`https://finnhub.io/api/v1/stock/profile2`, {
                    params: { symbol, token: FINNHUB_API_KEY },
                });
                const metricResponse = await axios.get(`https://finnhub.io/api/v1/stock/metric`, {
                    params: { symbol, metric: "all", token: FINNHUB_API_KEY },
                });

                const metric = metricResponse.data.metric || {};
                const currentPrice = quoteResponse.data.c;
                const rawFiftyTwoWeekHigh = metric["52WeekHigh"] ?? null;
                const rawFiftyTwoWeekLow = metric["52WeekLow"] ?? null;

                const fiftyTwoWeekHigh = (rawFiftyTwoWeekHigh !== null && rawFiftyTwoWeekHigh >= currentPrice)
                    ? rawFiftyTwoWeekHigh
                    : null;

                const fiftyTwoWeekLow = rawFiftyTwoWeekLow ?? null;

                return {
                    symbol,
                    amount,
                    currentPrice,
                    fiftyTwoWeekHigh,
                    fiftyTwoWeekLow,
                    companyName: profileResponse.data.name,
                    companyDescription: profileResponse.data.finnhubIndustry,
                    businessSummary: profileResponse.data.weburl
                        ? `More at ${profileResponse.data.weburl}`
                        : "",
                };
            })
        );

        const prompt = `
You are a professional investment advisor.

The user has invested in the following stocks:

${stockData
    .map(
        (stock) => `
### ${stock.companyName} (${stock.symbol})

- **Sector:** ${stock.companyDescription}
- **Business Focus:** Briefly describe what the company sells or produces in one clear sentence.
- **Current Stock Price:** $${stock.currentPrice}
- **52-Week High:** ${stock.fiftyTwoWeekHigh !== null ? `$${stock.fiftyTwoWeekHigh}` : "Data unavailable"}
- **52-Week Low:** ${stock.fiftyTwoWeekLow !== null ? `$${stock.fiftyTwoWeekLow}` : "Data unavailable"}
`
    )
    .join("\n")}

---

### User's Investment Goals:
${goals.map((goal, i) => `${i + 1}. ${goal}`).join("\n")}

---

### Your Task:

- **Part 1: Company Overview**
  - For each company:
    - Very briefly describe the sector and what the company produces or sells (maximum 2 short sentences).

- **Part 2: Stock Information**
  - For each stock:
    - Show:
      - **Current Stock Price** ($)
      - **52-Week High** ($)
      - **52-Week Low** ($)

- **Part 3: Investment Advice**
  - If the user's goals seem unrealistic or too high:
    - Show only this single sentence and end the response: 
      > **"This goal may be unrealistic or require a very long time to achieve."**
    - Do not provide any BUY, SELL, or HOLD advice.
  - Otherwise:
    - For each stock:
      - Recommend: **BUY**, **SELL**, or **HOLD** based on the current stock data and user's goals.
      
- Keep the language professional, well-structured with clear sections and bullet points.
- Always show actual numerical values where applicable.
`;

        const gptResponse = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [{ role: "user", content: prompt }],
        });

        res.status(200).json({ invest: gptResponse.choices[0].message.content });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Investment analysis failed." });
    }
});

app.post("/addNote", isAuthenticated, (req, res) => {
    const { title, content } = req.body;

    if (!title || !content) {
        console.error("Missing title or content");
        return res.status(400).json({ error: "Title and content are required" });
    }

    const userId = req.session.user.id;
    console.log("Saving note for user:", userId);

    const sql = "INSERT INTO notes (user_id, title, content) VALUES (?, ?, ?)";
    db.query(sql, [userId, title, content], (err, result) => {
        if (err) {
            console.error("Error saving note:", err);
            return res.status(500).json({ error: "Failed to save note" });
        }
        console.log("Note saved successfully:", result);
        res.status(201).json({ message: "Note saved successfully" });
    });
});

app.get("/getNotes", isAuthenticated, (req, res) => {
    const userId = req.session.user.id;

    const sql = "SELECT id, title, content, created_at FROM notes WHERE user_id = ?";
    db.query(sql, [userId], (err, results) => {
        if (err) {
            console.error("Error fetching notes:", err);
            return res.status(500).json({ error: "Failed to fetch notes" });
        }
        res.status(200).json({ notes: results });
    });
});

app.post("/addArticle", isAuthenticated, upload.array("images", 4), (req, res) => {
    const { title, body } = req.body;
    const userId = req.session.user.id;
    const images = req.files.map((file) => `/uploads/${file.filename}`);

    if (!title || !body) {
        return res.status(400).json({ error: "Title and body are required" });
    }

    const sqlInsert = "INSERT INTO articles (user_id, title, body, images) VALUES (?, ?, ?, ?)";
    db.query(sqlInsert, [userId, title, body, JSON.stringify(images)], (err, result) => {
        if (err) {
            console.error("Error inserting article:", err);
            return res.status(500).json({ error: "Failed to add article" });
        }

        res.status(201).json({ message: "Article added successfully" });
    });
});

app.use("/uploads", express.static("uploads"));// szimame snimkite ot papkata

app.get("/getArticles", (req, res) => {
    const sql = "SELECT id, title, body, images, created_at FROM articles";
    db.query(sql, (err, results) => {
        if (err) {
            console.error("Error fetching articles:", err);
            return res.status(500).json({ error: "Failed to fetch articles" });
        }
        res.status(200).json({ articles: results });
    });
});

app.delete("/deleteArticle/:id", isAuthenticated, (req, res) => {
    const articleId = req.params.id;
    const userId = req.session.user.id;

    const sqlSelect = "SELECT images FROM articles WHERE id = ? AND user_id = ?";
    db.query(sqlSelect, [articleId, userId], (err, results) => {
        if (err) {
            console.error("Error fetching article images:", err);
            return res.status(500).json({ error: "Failed to fetch article images" });
        }

        if (results.length === 0) {
            console.error("Article not found or access denied:", articleId);
            return res.status(404).json({ error: "You do not have permission to delete this article" });
        }

        let images = [];
        try {
            const imagesField = results[0].images;


            if (typeof imagesField === 'string' && imagesField.trim() !== "") {
                images = JSON.parse(imagesField);
                if (!Array.isArray(images)) images = [];
            } else if (Array.isArray(imagesField)) {
                images = imagesField;
            }
        } catch (parseError) {
            console.error("Error parsing images JSON:", parseError);
            images = [];
        }
        images.forEach((imagePath) => {
            const fullPath = path.join(__dirname, imagePath);
            fs.stat(fullPath, (err, stats) => {
                if (err) {
                    console.error("Error checking file existence:", err);
                    return;
                }
                if (stats.isFile()) {
                    fs.unlink(fullPath, (err) => {
                        if (err) {
                            console.error("Error deleting image:", err);
                        } else {
                            console.log("Deleted image:", fullPath);
                        }
                    });
                } else {
                    console.warn("File is not a valid file:", fullPath);
                }
            });
        });

        const sqlDelete = "DELETE FROM articles WHERE id = ? AND user_id = ?";
        db.query(sqlDelete, [articleId, userId], (err, result) => {
            if (err) {
                console.error("Error deleting article:", err);
                return res.status(500).json({ error: "Failed to delete article" });
            }

            console.log("Article deleted successfully:", articleId);
            res.status(200).json({ message: "Article deleted successfully" });
        });
    });
});

app.post("/addLink", isAuthenticated, uploadLinkImage.single("image"), (req, res) => {
    const { title, description, url } = req.body;
    const userId = req.session.user.id;
    const imagePath = req.file ? `/uploads/links/${req.file.filename}` : null;

    const sql = "INSERT INTO links (user_id, title, description, url, image_path) VALUES (?, ?, ?, ?, ?)";
    db.query(sql, [userId, title, description, url, imagePath], (err, result) => {
        if (err) {
            console.error("Error adding link:", err);
            return res.status(500).json({ error: "Failed to add link" });
        }
        res.status(201).json({ message: "Link added successfully" });
    });
});


app.get("/getLinks", isAuthenticated, (req, res) => {
    const sql = "SELECT * FROM links ORDER BY created_at DESC";
    db.query(sql, (err, results) => {
        if (err) {
            console.error("Error fetching links:", err);
            return res.status(500).json({ error: "Failed to fetch links" });
        }
        res.status(200).json({ links: results });
    });
});


app.delete("/deleteLink/:id", isAuthenticated, (req, res) => {
    const linkId = req.params.id;
    const { imagePath } = req.body;
    const userId = req.session.user.id;


    const sqlDelete = "DELETE FROM links WHERE id = ? AND user_id = ?";
    db.query(sqlDelete, [linkId, userId], (err, result) => {
        if (err) {
            console.error("Error deleting link:", err);
            return res.status(500).json({ error: "Failed to delete link" });
        }

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Link not found or access denied" });
        }


        if (imagePath) {
            const fullPath = path.join(__dirname, imagePath);
            fs.unlink(fullPath, (err) => {
                if (err) {
                    console.error("Error deleting image:", err);
                }
            });
        }

        res.status(200).json({ message: "Link deleted successfully" });
    });
});

app.get("/api/news", async (req, res) => {
    try {
        const response = await axios.get(
            `https://newsapi.org/v2/everything?q=finance&apiKey=${process.env.NEWSAPI_KEY}&pageSize=20&language=en`
        );
        // frontend format
        const formattedNews = response.data.articles.map(article => ({
            uuid: article.url.hashCode(), // ID
            title: article.title,
            description: article.description,
            url: article.url,
            image_url: article.urlToImage || '/default-news-image.jpg',
            source: article.source.name,
            published_at: article.publishedAt
        }));

        res.status(200).json({
            success: true,
            data: formattedNews
        });

    } catch (error) {
        console.error("News API error:", error);

        const fallbackNews = [
            {
                uuid: 1,
                title: "Financial Markets Update",
                description: "Latest updates from global financial markets",
                url: "https://example.com/finance-news",
                image_url: "https://via.placeholder.com/300x200?text=Finance+News",
                source: "Financial Times",
                published_at: new Date().toISOString()
            },
        ];

        res.status(200).json({
            success: false,
            message: "Using fallback news data",
            data: fallbackNews
        });
    }
});

String.prototype.hashCode = function () {
    let hash = 0;
    for (let i = 0; i < this.length; i++) {
        hash = ((hash << 5) - hash) + this.charCodeAt(i);
        hash |= 0;
    }
    return hash;
};

app.post("/save-stock-advice", (req, res) => {
    const { advice } = req.body;

    if (!advice || advice.trim().length === 0) {
        return res.status(400).json({ error: "Advice is required" });
    }

    const sql = "INSERT INTO stockadvisorAi (response) VALUES (?)";
    db.query(sql, [advice], (err, result) => {
        if (err) {
            console.error("Error saving advice:", err);
            return res.status(500).json({ error: "Failed to save advice" });
        }
        res.status(201).json({ message: "Advice saved successfully" });
    });
});

app.post("/save-investment-advice", (req, res) => {
    const { advice } = req.body;

    if (!advice) {
        return res.status(400).json({ error: "Advice is required" });
    }

    const sql = "INSERT INTO investadvisorAi (response) VALUES (?)";
    db.query(sql, [advice], (err, result) => {
        if (err) {
            console.error("Error saving advice:", err);
            return res.status(500).json({ error: "Failed to save advice" });
        }
        res.status(201).json({ message: "Advice saved successfully" });
    });
});

app.get("/get-stock-advice", isAuthenticated, (req, res) => {
    const sql = "SELECT id, response AS fullResponse, LEFT(response, 100) AS summary FROM stockadvisorAi";
    db.query(sql, (err, results) => {
        if (err) {
            console.error("Error fetching stock advice:", err);
            return res.status(500).json({ error: "Failed to fetch stock advice" });
        }
        res.status(200).json({ responses: results });
    });
});

app.get("/get-investment-advice", isAuthenticated, (req, res) => {
    const sql = "SELECT id, response AS fullResponse, LEFT(response, 100) AS summary FROM investadvisorAi";
    db.query(sql, (err, results) => {
        if (err) {
            console.error("Error fetching investment advice:", err);
            return res.status(500).json({ error: "Failed to fetch investment advice" });
        }
        res.status(200).json({ responses: results });
    });
});

app.post("/save-budget-plan", (req, res) => {
    const { plan } = req.body;
    console.log("Received plan:", plan);

    if (!plan || plan.trim().length === 0) {
        return res.status(400).json({ error: "Budget plan is required" });
    }

    const sql = "INSERT INTO budgetplannerAi (response) VALUES (?)";
    db.query(sql, [plan], (err, result) => {
        if (err) {
            console.error("Error saving budget plan:", err);
            return res.status(500).json({ error: "Failed to save budget plan" });
        }
        res.status(201).json({ message: "Budget plan saved successfully" });
    });
});

app.get("/get-budget-planner", isAuthenticated, (req, res) => {
    const sql = "SELECT id, response AS fullResponse, LEFT(response, 100) AS summary FROM budgetplannerAi";
    db.query(sql, (err, results) => {
        if (err) {
            console.error("Error fetching budget plans:", err);
            return res.status(500).json({ error: "Failed to fetch budget plans" });
        }
        res.status(200).json({ responses: results });
    });
});

app.post("/logout", (req, res) => {
    if (req.session) {
        req.session.destroy((err) => {
            if (err) {
                return res.status(500).json({ error: "Failed to log out" });
            } else {
                res.clearCookie("connect.sid"); // izchistwane na biskwitkite
                return res.status(200).json({ message: "Logout successful" });
            }
        });
    } else {
        res.status(400).json({ error: "No active session" });
    }
});



// -------------------- Server Start --------------------

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});












