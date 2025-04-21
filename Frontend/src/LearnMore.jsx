import React from "react";
import Header from "./assets/header";
import Footer from "./assets/footer";

function LearnMore() {
  return (
    <>
      <Header />
      <div className="pt-24 min-h-screen bg-gradient-to-br from-blue-600 to-blue-200 p-10">
        <h1 className="text-white text-4xl font-bold mb-10 text-center mt-10">
          Learn More About Our Site
        </h1>
        <div className="bg-white max-w-5xl mx-auto p-8 rounded-2xl shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            Overview
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            Welcome to our financial platform! This project is designed to help you take control of your finances, make informed decisions, and achieve your financial goals. Whether you're looking to calculate your loan payments, get investment advice, or learn more about financial planning, our platform has everything you need in one place.
          </p>

         

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            AI Tools
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            Our platform includes advanced AI tools to provide personalized advice and insights tailored to your financial needs. Here's what each AI tool does:
          </p>
          <ul className="list-disc list-inside text-gray-700 text-lg mb-6 text-left">
            <li>
              <strong>Investment AI:</strong> This tool helps you make informed investment decisions by analyzing your financial goals, available funds, and risk tolerance. It provides recommendations on how to allocate your resources effectively.
            </li>
            <li>
              <strong>Budgeting AI:</strong> Designed to help you manage your monthly expenses, this AI tool analyzes your income and spending habits to suggest ways to save more and achieve your financial goals faster.
            </li>
            <li>
              <strong>Stock Advisor AI:</strong> This AI tool provides insights into stock market trends and suggests potential investment opportunities based on your preferences and risk appetite.
            </li>
          </ul>

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            Calculators
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            Our calculators are designed to simplify complex financial calculations. Whether you're planning a loan, analyzing investments, or converting currencies, our tools provide accurate and reliable results. Here's a detailed breakdown of each calculator:
          </p>
          <ul className="list-disc list-inside text-gray-700 text-lg mb-6 text-left">
            <li>
              <strong>Investment Calculator:</strong> This tool helps you analyze your stock or ETF returns. By entering details like the purchase price, current price, dividends, and the number of shares, you can calculate your total profit, dividend income, and percentage return. This calculator is perfect for investors who want to track their portfolio performance.
            </li>
            <li>
              <strong>Loan Calculator:</strong> Planning to take a loan? Our Loan Calculator helps you calculate your monthly payments and total interest based on the loan amount, interest rate, and repayment period. This tool is ideal for budgeting and understanding the financial commitment of a loan.
            </li>
            <li>
              <strong>Currency Calculator:</strong> Need to convert currencies? Our Currency Calculator uses real-time exchange rates to provide accurate conversions. You can also filter by region or sort by rate, making it a versatile tool for travelers, businesses, or anyone dealing with multiple currencies.
            </li>
          </ul>

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            Notes
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            The Notes feature allows you to save and manage important financial information directly on your profile. Whether it's a reminder about an upcoming payment or a note about your investment strategy, you can easily access and organize your notes. Notes are securely stored and accessible only to you.
          </p>

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            Articles
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            The **Articles** section is a hub for financial news, trends, and insights. It provides users with up-to-date information on topics such as global markets, cryptocurrency, real estate, and more. Articles are curated to help users stay informed about the latest developments in the financial world. 
          </p>
          <p className="text-gray-700 text-lg mb-6 text-left">
            Users can also contribute their own articles, making this section a collaborative space for sharing knowledge. Each article includes a title, description, and optional images to make the content more engaging. Administrators have the ability to manage articles, ensuring the quality and relevance of the content.
          </p>

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            Useful Links
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            The **Useful Links** section is a curated library of resources to help users learn more about financial planning, investment strategies, and budgeting tips. Users can explore links to trusted websites, tools, and guides that provide valuable insights into managing finances effectively.
          </p>
          <p className="text-gray-700 text-lg mb-6 text-left">
            Users can also add their own links to create a personalized collection of resources. Each link includes a title, description, and optional image for better organization. This feature is perfect for users who want to keep all their favorite financial resources in one place.
          </p>

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            About Us
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            The **About Us** page introduces the team behind this platform you can see this here <a href="/about" className="text-blue-800 underline">About us</a>. It highlights the contributions of each team member, showcasing their expertise and dedication to creating a user-friendly financial platform. The page also outlines the mission and vision of the project, emphasizing the goal of making financial management accessible to everyone.
          </p>
          <p className="text-gray-700 text-lg mb-6 text-left">
            Users can learn about the team’s journey, the challenges they faced, and the innovative solutions they developed to bring this platform to life. The page also includes images and detailed descriptions of each team member’s role and achievements.
          </p>

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            Why Choose Us?
          </h2>
          <p className="text-gray-700 text-lg mb-6 text-left">
            Our platform is designed with you in mind. We aim to make financial management accessible and straightforward for everyone. Whether you're a beginner or an experienced investor, our tools and resources are here to support you every step of the way.
          </p>

          <h2 className="text-2xl font-bold mb-4 text-center text-blue-800">
            Get Started
          </h2>
          <p className="text-gray-700 text-lg text-left">
            Ready to take control of your finances? Create an account today and explore all the features our platform has to offer. If you have any questions, visit our <a href="/faq" className="text-blue-800 underline">FAQ</a> page or contact us directly. We're here to help!
          </p>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default LearnMore;