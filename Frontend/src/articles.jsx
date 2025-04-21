import React, { useEffect, useState } from "react";
import Header from "./assets/header";
import Footer from "./assets/footer";
import axios from "axios";

function News() {
    const [news, setNews] = useState([]);
    const [selectedArticle, setSelectedArticle] = useState(null);

    useEffect(() => {
        const fetchNews = async () => {
            try {
                const response = await axios.get("http://localhost:8000/api/news"); // Backend proxy
                setNews(response.data.data);
            } catch (error) {
                console.error("Error fetching news:", error);
            }
        };

        fetchNews();
    }, []);

    return (
        <>
            <Header />
            <div className="min-h-screen bg-gray-100 p-6">
                <h1 className="text-4xl font-bold text-center text-blue-800 mb-10">
                    Latest Financial News
                </h1>

                {selectedArticle ? (
                    <div className="bg-white p-6 rounded-lg shadow-lg max-w-4xl mx-auto">
                        <button
                            onClick={() => setSelectedArticle(null)}
                            className="text-blue-600 hover:underline mb-4"
                        >
                            ← Back to News
                        </button>
                        <h2 className="text-2xl font-bold mb-4">{selectedArticle.title}</h2>
                        <img
                            src={selectedArticle.image_url}
                            alt={selectedArticle.title}
                            className="w-full h-64 object-cover rounded-lg mb-4"
                        />
                        <p className="text-gray-700 text-lg">{selectedArticle.description}</p>
                        <a
                            href={selectedArticle.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:underline mt-4 block"
                        >
                            Read Full Article
                        </a>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                        {news.map((article) => (
                            <div
                                key={article.uuid}
                                className="bg-white p-4 rounded-lg shadow-lg cursor-pointer hover:scale-105 transition-transform"
                                onClick={() => setSelectedArticle(article)}
                            >
                                <img
                                    src={article.image_url}
                                    alt={article.title}
                                    className="w-full h-40 object-cover rounded-lg mb-4"
                                />
                                <h3 className="text-lg font-bold mb-2">{article.title}</h3>
                                <p className="text-gray-600 text-sm line-clamp-3">
                                    {article.description}
                                </p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
            <Footer />
        </>
    );
}

export default News;