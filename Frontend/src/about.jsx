import React from 'react';
import Header from './assets/header';
import Footer from './assets/footer';

function About() {
    return (
        <>
            <Header />
            <div className="p-5 pt-24">
                <h1 className="text-blue-500 text-4xl font-bold mb-10 mt-10 text-center">-- About Us --</h1>


                <div className="container mx-auto mb-16 bg-blue-400 p-5 rounded-lg shadow-lg">
                    <img
                        src="team.png"
                        alt="Our Team"
                        className="w-full h-64 object-cover rounded-lg shadow-lg mb-5"
                    />
                    <p className="text-black text-center text-lg">
                        <b>
                            We are a passionate team of developers, designers, and financial experts dedicated to helping you succeed.
                            Our goal is to provide you with the best tools and resources to take control of your finances and achieve your goals.
                            Together, we strive to innovate and create solutions that make budgeting and financial planning accessible to everyone.
                        </b>
                    </p>
                </div>

                <div className="container mx-auto mb-16 flex items-center bg-blue-400 p-5 rounded-lg shadow-lg">
                    <img
                        src="imgqvor.png"
                        alt="Our Mission"
                        className="w-1/3 h-64 object-cover rounded-lg shadow-lg mr-5"
                    />
                    <div className="text-left">
                        <h2 className="text-2xl font-bold mb-3">Yavor Penev</h2>
                        <p className="text-black text-lg">
                            Every team needs someone who willingly dives headfirst into the back end of things, and for us, that's
                             Yavor. He’s our database whisperer—making sure data flows like a well-behaved spreadsheet. He laid t
                             he foundations for the currency calculator (because who doesn’t love watching money convert), created 
                             the notes feature for those who like to jot things down mid-budget crisis, and built the entire user 
                             article section. Oh, and if you've seen users giving advice to other users—yep, that’s him too. A man
                              of many hats… all of them technical.
                        </p>

                    </div>
                </div>


                <div className="container mx-auto mb-16 flex items-center bg-blue-400 p-5 rounded-lg shadow-lg">
                    <img
                        src="imgvelinov.png"
                        alt="Our Vision"
                        className="w-1/3 h-64 object-cover rounded-lg shadow-lg mr-5"
                    />
                    <div className="text-left">
                        <h2 className="text-2xl font-bold mb-3">Kristian Velinov</h2>
                        <p className="text-black text-lg">
                        Kristian is the reason the site doesn't look like it was designed in 2005. As the lead design developer,
                         he crafted the homepage, loan calculator, profile page, and basically anything else with a pixel on it. 
                         If you like how the buttons click, how the layout flows, or how everything somehow just feels right, tha
                         t’s Kristian’s doing. He didn’t just work on the design—he became the design. We haven’t confirmed this,
                          but he may dream in Tailwind CSS.
                        </p>

                    </div>
                </div>


                <div className="container mx-auto mb-16 flex items-center bg-blue-400 p-5 rounded-lg shadow-lg">
                    <img
                        src="imgmih.png"
                        alt="Our Team"
                        className="w-1/3 h-64 object-cover rounded-lg shadow-lg mr-5"
                    />
                    <div className="text-left">
                        <h2 className="text-2xl font-bold mb-3">Kristiqn Mihailov</h2>
                        <p className="text-black text-lg">
                        If something magical is happening behind the scenes, it's probably Kristiyan. He’s our main backend 
                        developer and the mastermind behind the AI features, making sure our smart tools actually, well… act
                         smart. He handles all the training, data wrangling, and technical wizardry that turns raw information
                          into helpful advice. Think of him as the guy teaching the robots how to help you budget better—no evi
                          l AI overlords here, just clean code and clever functionality.
                        </p>
                    </div>
                </div>


                <div className="container mx-auto mb-16 flex items-center bg-blue-400 p-5 rounded-lg shadow-lg">
                    <img
                        src="img77.png"
                        alt="Our Values"
                        className="w-1/3 h-64 object-cover rounded-lg shadow-lg mr-5"
                    />
                    <div className="text-left">
                        <h2 className="text-2xl font-bold mb-3">Borislav Stoinev</h2>
                        <p className="text-black text-lg">
                        Bobi handled the less glamorous but no less essential parts of the site. He built the Learn More page
                        , FAQ, and Legal section—because someone has to keep things legit (and occasionally remind users we a
                        re not responsible for bad crypto investments). He also helped lay the foundation for the profile page
                         and created the News section, because financial education shouldn’t come without a bit of context
                        </p>
                    </div>
                </div>


                <div className="container mx-auto flex items-center bg-blue-400 p-5 rounded-lg shadow-lg">
                    <img
                        src="imgdein.png"
                        alt="Our Commitment"
                        className="w-1/3 h-64 object-cover rounded-lg shadow-lg mr-5"
                    />
                    <div className="text-left">
                        <h2 className="text-2xl font-bold mb-3">Deyan Naidenov</h2>
                        <p className="text-black text-lg">
                        Deyan may not have touched every corner of the project, but what he did touch, he upgraded. He bui
                        lt the compact pop-up calculator—perfect for quick math panics—and implemented email verification 
                        to keep spam bots out and real users in. Not the flashiest tasks, but hey, someone has to do the s
                        tuff that actually makes the site function smoothly without anyone noticing.
                        </p>
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
}

export default About;