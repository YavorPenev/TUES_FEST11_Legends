import React, { useState, useEffect } from "react";
import Header from "./assets/header";
import Footer from "./assets/footer";
import axios from "axios";

function PerArticles() {
    const [articles, setArticles] = useState([]);
    const [showForm, setShowForm] = useState(false);
    const [newArticle, setNewArticle] = useState({
        title: "",
        body: "",
        images: [],
    });

    // Зареждане на всички статии
    useEffect(() => {
        const fetchArticles = async () => {
            try {
                const response = await axios.get("http://localhost:8000/getArticles");
                setArticles(response.data.articles);
            } catch (error) {
                console.error("Failed to fetch articles:", error);
            }
        };

        fetchArticles();
    }, []);

    // Добавяне на нова статия
    const handleAddArticle = async () => {
        const formData = new FormData();
        formData.append("title", newArticle.title);
        formData.append("body", newArticle.body);
        newArticle.images.forEach((image) => {
            formData.append("images", image);
        });

        try {
            const response = await axios.post("http://localhost:8000/addArticle", formData, {
                headers: { "Content-Type": "multipart/form-data" },
                withCredentials: true,
            });
            alert(response.data.message);
            setArticles([...articles, { ...newArticle, id: Date.now() }]);
            setShowForm(false);
            setNewArticle({ title: "", body: "", images: [] });
        } catch (error) {
            console.error("Failed to add article:", error);
            alert("Failed to add article. Please try again.");
        }
    };

    // Обработка на качване на снимки
    const handleImageUpload = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 4) {
            alert("You can upload up to 4 images.");
            return;
        }
        setNewArticle({ ...newArticle, images: files });
    };

    // Функция за обработка на текста и замяна на >текст< с линкове
    const processBodyText = (text) => {
        return text.replace(/>(.*?)</g, (match, p1) => {
            return `<a href="${p1}" target="_blank" rel="noopener noreferrer" class="text-blue-500 hover:underline">link here</a>`;
        });
    };

    return (
        <>
            <Header />
            <div className="p-6 pt-14 bg-gray-100 min-h-screen">
                <h1 className="text-3xl font-bold mb-6">Articles</h1>

                <div className="space-y-6">
                    {articles.map((article) => (
                        <div
                            key={article.id}
                            className="bg-white p-6 rounded-lg shadow-lg flex flex-col space-y-4"
                        >
                            <h2 className="text-2xl font-bold">{article.title}</h2>
                            <p
                                className="text-gray-700 text-justify indent-4"
                                dangerouslySetInnerHTML={{ __html: processBodyText(article.body) }}
                            ></p>
                            <div className="grid grid-cols-4 gap-4">
                                {article.images.map((image, index) => (
                                    <img
                                        key={index}
                                        src={`http://localhost:8000${image}`} // Зареждане на снимката от сървъра
                                        alt={`Article ${article.id} Image ${index + 1}`}
                                        className="w-full h-32 object-cover rounded"
                                    />
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                <button
                    onClick={() => setShowForm(true)}
                    className="fixed  bottom-10 right-10 bg-blue-500 text-white p-4 rounded-2xl shadow-lg text-4xl font-bold flex items-center justify-center"
                >
                    +
                </button>

                {showForm && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                        <div className="bg-white p-6 rounded-lg shadow-lg w-96">
                            <h2 className="text-xl font-bold mb-4">Add New Article</h2>
                            <input
                                type="text"
                                placeholder="Title"
                                value={newArticle.title}
                                onChange={(e) =>
                                    setNewArticle({ ...newArticle, title: e.target.value })
                                }
                                className="w-full p-2 border border-gray-300 rounded mb-4"
                            />
                            <textarea
                                placeholder="Body (use >link< to add clickable links)"
                                value={newArticle.body}
                                onChange={(e) =>
                                    setNewArticle({ ...newArticle, body: e.target.value })
                                }
                                className="w-full p-2 border border-gray-300 rounded mb-4"
                            />
                            <input
                                type="file"
                                multiple
                                accept="image/*"
                                onChange={handleImageUpload}
                                className="w-full p-2 border border-gray-300 rounded mb-4"
                            />
                            <div className="flex justify-between">
                                <button
                                    onClick={() => setShowForm(false)}
                                    className="bg-red-500 text-white px-4 py-2 rounded"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleAddArticle}
                                    className="bg-blue-500 text-white px-4 py-2 rounded"
                                >
                                    Save
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
            <Footer />
        </>
    );
}

export default PerArticles;