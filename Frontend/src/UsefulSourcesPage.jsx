import React, { useState, useEffect } from "react";
import Header from "./assets/header";
import Footer from "./assets/footer";
import axios from "axios";
import { FaPlus, FaTrash } from "react-icons/fa";

function Sitelinks() {
    const [links, setLinks] = useState([]);
    const [showForm, setShowForm] = useState(false);
    const [newLink, setNewLink] = useState({
        title: "",
        description: "",
        url: "",
        image: null
    });

    useEffect(() => {
        fetchLinks();
    }, []);

    const fetchLinks = async () => {
        try {
            const response = await axios.get("http://localhost:8000/getLinks", {
                withCredentials: true
            });
            setLinks(response.data.links);
        } catch (error) {
            console.error("Error fetching links:", error);
        }
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setNewLink(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleImageChange = (e) => {
        setNewLink(prev => ({
            ...prev,
            image: e.target.files[0]
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append("title", newLink.title);
        formData.append("description", newLink.description);
        formData.append("url", newLink.url);
        if (newLink.image) {
            formData.append("image", newLink.image);
        }

        try {
            await axios.post("http://localhost:8000/addLink", formData, {
                headers: {
                    "Content-Type": "multipart/form-data"
                },
                withCredentials: true
            });
            setShowForm(false);
            setNewLink({
                title: "",
                description: "",
                url: "",
                image: null
            });
            fetchLinks();
        } catch (error) {
            console.error("Error adding link:", error);
        }
    };

    const handleDelete = async (id, imagePath) => {
        try {
            await axios.delete(`http://localhost:8000/deleteLink/${id}`, {
                data: { imagePath },
                withCredentials: true
            });
            fetchLinks();
        } catch (error) {
            console.error("Error deleting link:", error);
        }
    };

    return (
        <>
            <Header />
            <div className=" min-h-screen bg-white p-6 mt-19">
                <div className="max-w-6xl mx-auto">
                    <h1 className="text-5xl font-bold text-gray-800 mb-8 mt-3">Links to useful sites</h1>

                    <div className="grid grid-cols-1 gap-6">
                        {links.map(link => (
                            <div key={link.id} className="bg-blue-300 rounded-3xl shadow-md overflow-hidden flex flex-col md:flex-row relative border-2">
                                <button
                                    onClick={() => handleDelete(link.id, link.image_path)}
                                    className="absolute top-2 right-2 text-red-500 hover:text-red-700 "
                                >
                                    <FaTrash />
                                </button>
                                {link.image_path && (
                                    <img
                                        src={`http://localhost:8000${link.image_path}`}
                                        alt={link.title}
                                        className="w-full md:w-1/2 h-[430px] object-cover"
                                    />
                                )}
                                <div className="p-8 flex-1 flex flex-col justify-between">
                                    <h2 className="text-3xl font-bold mb-6">{link.title}</h2>
                                    <div className="text-gray-800 text-left  leading-relaxed">
                                        <p>
                                            <span className="pl-8">{link.description.split('\n')[0]}</span>
                                        </p>
                                        {link.description.split('\n').slice(1).map((line, idx) => (
                                            <p key={idx}>{line}</p>
                                        ))}
                                    </div>

                                    <a
                                        href={link.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="mt-6 text-blue-500 hover:underline font-medium"
                                    >
                                        Visit Site here
                                    </a>
                                </div>
                            </div>
                        ))}
                    </div>

                    <button
                        onClick={() => setShowForm(true)}
                        className="fixed bottom-8 right-8 bg-blue-500 text-white rounded-full p-4 shadow-lg hover:bg-blue-600 transition"
                    >
                        <FaPlus size={24} />
                    </button>

                    {showForm && (
                        <div className=" flex  items-center justify-center p-4 z-50">
                            <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl">
                                <div className="p-6">
                                    <h2 className="text-2xl font-bold mb-4">Add New Resource</h2>
                                    <div className="flex flex-col md:flex-row gap-6">
                                        {/* Left: form */}
                                        <form onSubmit={handleSubmit} className="flex-1 space-y-4">
                                            <div>
                                                <label className="block text-gray-700 mb-2">Title</label>
                                                <input
                                                    type="text"
                                                    name="title"
                                                    value={newLink.title}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border rounded"
                                                    placeholder="Write the title here"
                                                    required
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-gray-700 mb-2">Description</label>
                                                <textarea
                                                    name="description"
                                                    value={newLink.description}
                                                    onChange={handleInputChange}
                                                    maxLength={624}
                                                    className="w-full px-3 py-2 border rounded h-36"
                                                    placeholder="Describe the resource..."
                                                    required
                                                />
                                                <p className="text-xs text-gray-500 text-right mt-1">
                                                    {newLink.description.length}/624 characters
                                                </p>
                                            </div>
                                            <div>
                                                <label className="block text-gray-700 mb-2">URL</label>
                                                <input
                                                    type="url"
                                                    name="url"
                                                    value={newLink.url}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border rounded"
                                                    placeholder="Example: https://example.com"
                                                    required
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-gray-700 mb-2">Image (Optional)</label>
                                                <input
                                                    type="file"
                                                    accept="image/*"
                                                    onChange={handleImageChange}
                                                    className="w-full px-3 py-2 border rounded"
                                                />
                                            </div>
                                            <div className="flex justify-end space-x-4 mt-6">
                                                <button
                                                    type="button"
                                                    onClick={() => setShowForm(false)}
                                                    className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
                                                >
                                                    Cancel
                                                </button>
                                                <button
                                                    type="submit"
                                                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                                                >
                                                    Save
                                                </button>
                                            </div>
                                        </form>

                                        {/* Right: instructions */}
                                        <div className="w-full md:w-64 bg-gray-50 border rounded p-4 text-sm text-gray-700">
                                            <h3 className="font-semibold mb-2">How to add a link:</h3>
                                            <ul className="list-disc list-inside space-y-1">
                                                <li><strong>Title:</strong> Enter a short and clear title for the resource — for example, "React Docs" or "Free Design Assets".</li>
                                                <li><strong>Description:</strong> Add a brief explanation of what this resource is about, why it’s useful or when it can be used.</li>
                                                <li><strong>URL:</strong> Paste the full web address. Make sure it starts with <code>https://</code>, so the link works properly.</li>
                                                <li><strong>Image (optional):</strong> You can upload an image (like a logo or preview) to make your resource card more attractive.</li>
                                                <li>When ready, click <strong>Save</strong>. The resource will appear immediately in the list below.</li>
                                                <li>If you change your mind, hit <strong>Cancel</strong> to close this window without saving.</li>
                                            </ul>
                                            <p className="mt-3 text-xs text-gray-500">
                                                Tip: Keep titles and descriptions short and to the point for better readability.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <Footer />
        </>
    );
}

export default Sitelinks;