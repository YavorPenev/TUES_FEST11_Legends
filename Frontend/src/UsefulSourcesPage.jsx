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
            <div className="min-h-screen bg-gray-100 p-6 mt-14">
                <div className="max-w-6xl mx-auto">
                    <h1 className="text-3xl font-bold text-gray-800 mb-8">Useful Resources</h1>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {links.map(link => (
                            <div key={link.id} className="bg-white rounded-lg shadow-md overflow-hidden relative">
                                <button 
                                    onClick={() => handleDelete(link.id, link.image_path)}
                                    className="absolute top-2 right-2 text-red-500 hover:text-red-700"
                                >
                                    <FaTrash />
                                </button>
                                <div className="p-4">
                                    <h2 className="text-xl font-semibold text-center mb-2">{link.title}</h2>
                                    <div className="flex items-start space-x-4">
                                        {link.image_path && (
                                            <img 
                                                src={`http://localhost:8000${link.image_path}`} 
                                                alt={link.title}
                                                className="w-24 h-24 object-cover rounded"
                                            />
                                        )}
                                        <p className="text-gray-600 flex-1">{link.description}</p>
                                    </div>
                                    <a 
                                        href={link.url} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="block mt-4 text-blue-500 hover:underline"
                                    >
                                        Visit Resource
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
                        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
                            <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl">
                                <div className="p-6">
                                    <h2 className="text-2xl font-bold mb-4">Add New Resource</h2>
                                    <form onSubmit={handleSubmit}>
                                        <div className="flex flex-col md:flex-row gap-6">
                                            <div className="flex-1">
                                                <div className="mb-4">
                                                    <label className="block text-gray-700 mb-2">Title</label>
                                                    <input
                                                        type="text"
                                                        name="title"
                                                        value={newLink.title}
                                                        onChange={handleInputChange}
                                                        className="w-full px-3 py-2 border rounded"
                                                        required
                                                    />
                                                </div>
                                                <div className="mb-4">
                                                    <label className="block text-gray-700 mb-2">Description</label>
                                                    <textarea
                                                        name="description"
                                                        value={newLink.description}
                                                        onChange={handleInputChange}
                                                        className="w-full px-3 py-2 border rounded h-32"
                                                        required
                                                    />
                                                </div>
                                                <div className="mb-4">
                                                    <label className="block text-gray-700 mb-2">URL</label>
                                                    <input
                                                        type="url"
                                                        name="url"
                                                        value={newLink.url}
                                                        onChange={handleInputChange}
                                                        className="w-full px-3 py-2 border rounded"
                                                        required
                                                    />
                                                </div>
                                            </div>
                                            <div className="flex-1">
                                                <div className="mb-4">
                                                    <label className="block text-gray-700 mb-2">Image (Optional)</label>
                                                    <input
                                                        type="file"
                                                        accept="image/*"
                                                        onChange={handleImageChange}
                                                        className="w-full px-3 py-2 border rounded"
                                                    />
                                                </div>
                                            </div>
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