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

    const handleImageUpload = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 4) {
            alert("You can upload up to 4 images.");
            return;
        }
        setNewArticle({ ...newArticle, images: files });
    };

    const processBodyText = (text) => {
        return text.replace(/>(.*?)</g, (match, p1) => {
            return `<a href="${p1}" target="_blank" rel="noopener noreferrer" class="text-blue-500 hover:underline">link here</a>`;
        });
    };

    const handleDeleteArticle = async (id) => {
        const confirmDelete = window.confirm("Are you sure you want to delete this article?");
        if (!confirmDelete) return;

        try {
            const response = await axios.delete(`http://localhost:8000/deleteArticle/${id}`, {
                withCredentials: true,
            });
            alert(response.data.message);
            setArticles(articles.filter((article) => article.id !== id));
        } catch (error) {
            if (error.response && error.response.status === 403) {
                alert("You do not have permission to delete this article.");
            } else {
                console.error("Failed to delete article:", error);
                alert("Failed to delete article. Please try again.");
            }
        }
    };


    return (
        <>
            <Header />
            <div className="p-6 pt-14 bg-gray-100 min-h-screen">
            <h1 className="text-5xl font-bold text-gray-800 mb-4 mt-15">Useful Articles</h1>

                <div className="space-y-6 ">
                    {articles.map((article) => (
                        <div
                            key={article.id}
                            className="bg-blue-200 p-6 rounded-lg shadow-lg flex flex-col space-y-4 relative"
                        >
                            <button
                                onClick={() => handleDeleteArticle(article.id)}
                                className="absolute top-2 right-2 text-red-500 hover:text-red-700 text-xl font-bold"
                            >
                                ✕
                            </button>
                            <h2 className="text-2xl font-bold">{article.title}</h2>
                            <p
                                className="text-gray-700 text-justify indent-4 break-words max-w-full"
                                dangerouslySetInnerHTML={{ __html: processBodyText(article.body) }}
                            ></p>
                            <div className="grid grid-cols-4 gap-4">
                                {article.images.map((image, index) => (
                                    <img
                                        key={index}
                                        src={`http://localhost:8000${image}`}
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
                    className="fixed bottom-10 right-10 bg-blue-500 text-white w-12 h-12 rounded-2xl shadow-lg text-3xl font-bold flex items-center justify-center"
                >
                    +
                </button>

                {showForm && (
                    <div className="p-6 pt-14 bg-transparent min-h-screen ">
                        <div className="bg-white p-6 rounded-lg shadow-lg w-4/4 h-4/4 flex">
                  
                            <div className="w-2/3 pr-4 border-r border-gray-300">
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
                                    className="w-full p-2 border border-gray-300 rounded mb-4 h-40"
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

                       
                            <div className="w-1/3 pl-4">
                                <h2 className="text-lg font-bold mb-2">How to Create an Article:</h2>
                                <ul className="list-disc list-inside text-sm text-gray-700 mb-4">
                                    <li>Enter a title in the "Title" field.</li>
                                    <br></br>
                                    <li>Write the content of the article in the "Body" field.</li>
                                    <br></br>
                                    <li>Use {">link address<"} to add clickable links.</li>
                                    <br></br>
                                    <li>Only administrator accounts can create articles.</li>
                                </ul>
                                <p className="text-sm text-gray-600">
                                    Make sure to upload up to a maximum of 4 images if needed. Images should be relevant to the article content.
                                </p>
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