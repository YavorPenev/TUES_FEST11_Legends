import React from "react";
import { Link } from "react-router";

function Footer() {
    return (
        <footer className="absolute w-full flex bg-blue-800 bg-gradient-to-b from-transparent to-gray-800 text-blue-100 p-5 justify-evenly flex-wrap items-center">
            <p>©legends Development Team</p>
            <Link className="hover:underline" to="/learnmore">Learn More Page</Link>
            <Link className="hover:underline" to="/articles">Articles Page</Link>
            <Link className="align-baseline hover:underline" to="/faq">FAQ Page</Link>
            <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
                <a
                    href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                    target="_blank"
                    rel="noopener noreferrer"
                ><img className="aspect-auto h-15" src="/youtube.png" alt="YouTube" /></a>
                <p>+39 06 6988 4857</p>
                <p>yavorpen@gmail.com</p>
            </div>
            <Link className="hover:underline" to="/stockai">To Stocks AI Assistant</Link>
            <Link to="/about" className="hover:underline">About</Link>
        </footer>
    );
}

export default Footer;