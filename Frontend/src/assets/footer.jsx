import React from "react";
import { Link } from "react-router";

function Footer() {
    return (
        <footer className="flex bg-blue-800 bg-gradient-to-b from-transparent to-gray-800 text-blue-100 p-5 justify-evenly flex-wrap items-center">
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
          <a
            href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img className="aspect-auto h-15" src="/youtube.png" alt="YouTube" />
          </a>
          <p>Copyright @2025</p>
          <p>©legends Development Team</p>
        </div>
        <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
          <p>legends@gmail.com</p>
          <p>+39 06 6988 4857</p>
          <p>+39 04 5355 9832</p>
        </div>
        <div className="flex flex-col gap-1 items-center">
        <Link to="/investcalc" className="text-blue-100 hover:underline">Investment Calculator</Link>
          <Link to="/calcloan" className="text-blue-100 hover:underline">Loan Calculator</Link>
          <Link to="/CurrencyCalculator" className="text-blue-100 hover:underline">Currency Calculator</Link>
        </div>
        <div className="flex flex-col gap-1 items-start">
          <Link to="/articles" className="text-blue-100 hover:underline">Articles</Link>
          <Link to="/about" className="text-blue-100 hover:underline">About Us</Link>
          <Link to="/law" className="text-blue-100 hover:underline">Terms of Use</Link>
        </div>
      </footer>
    );
}

export default Footer;