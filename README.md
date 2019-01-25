# Wiwasolvet Total Primary Energy Solutions
<h3>Prospecting Met Masts: What are they?</h3>

<p>Note: Presently this repo will generate the Wind Energy Report from a local/remote database. Will update the code to get API weather data retrieval and database table creation working.</p>
<h4>Intent</h4>
<p>This documentation is intended to help others install this free open source software, along with dependencies and get you started creating your own Prospecting Met Masts (PMM) reports.</p>

<h4>Purpose:</h4>
<p>I developed this new category of Met Mast to make data interpretation easier for decision makers and in an industry standard format. PMMs are similar in style to the more well-known Virtual Met Masts (VMM). VMMs include microsite analysis with calculated wind turbulence from obstacles and topography in close proximity that affects electricity production.</p>
<p>It is important to consider that PMMs are not the same as site based physical Met Masts - which provide high resolution data that records local turbulence, but only on short timescales over a year or two until developers have enough information to process and plan locations of wind turbines. This planning involves experts choosing from ideal optimizations based on electricity production potential and overall costs of the project amongst environmental, social and other factors.</p>

<h4>Highlights:</h4>
<ul>
<li>To encourage assessing a wide selection of potential energy locations.</li>
<li>Providing fine-tuned time-scale resolution at 1 hour steps over 30-60 years.</li>
<li>Provide the ability to include 10 years of forecasted hourly data.</li>
<li>Estimate electricity production based on prior weather-climate data, matched to a variety of wind turbines selected.</li>
<li>Help the small wind energy market and decentralized energy with a platform that promotes energy independence and interdependence at the appropriate scales.</li>
</ul>

<h4>Future Features:</h4>
<ul>
<li>Energy storage sizing capabilities</li>
<li>Solar energy estimations based on location and cloud cover.</li>
<li>Wind power availability modified by weather parameters:
<ul>
<li>Temperature, air pressure, humidity, precipitation (blade fouling), and elevation all affect energy content in the wind, beyond that of wind speed alone.</li>
</ul>
</li>
<li>Multiprocess package upgrade to enable faster data download speeds.</li>
<li>Code profiling/refactoring to provide processing speed improvements.</li>
<li>Multicore enabled option for simultaneous downloading of multiple chunks (years) of hourly data and as many locations that server/system can handle.</li>
<li>Integration with the MapBox platform to provide the ability to select locations visually and analyze relevant wind data intelligence to make the best decisions. HexGrid may be utilized.</li>
<li>MERRA2 data integration for free average annual wind histogram.</li>
<li>Darksky.io real-time and historical wind farm energy production, depending on daily data costs.</li>
<li>Docker enabled "3 server" unit-testing process for quality control and raw data validation.
<ul>
<li>Local MapBox server: to serve mapping tiles to MapBox map for localhost viewing.</li>
<li>Wiwasolvet server: to download and process data, and host website interface using WordPress/Drupal/Django etc.</li>
<li>Local DarkSky.io data server simulator: to create known data distributions, add potential data errors/unknowns, send data to Wiwasolvet server, and provide quality control within acceptable margin of error.</li>
<li>The above steps will help create a baseline for where and how much error is or can be related to: user input, and data processing.</li>
</ul>
</li>
<li>Separately a focus on data quality covered during the whitepaper-validation process, will analyze DarkSky.io data and performance based on known baseline Met Mast data.</li>
<li>Satellite image processing techniques/packages may be integrated eventually to provide an initial estimation of tree cover/obstacles as a guideline, and reduce user input error during site selection process.</li>
</ul>

<p>Community engagement is encouraged and I hope that people develop their own features, that Wiwasolvet may not have the time or capacity to develop in house.</p>

<p><strong>Author: Jacob Lawrence Thompson MSc. BASc.</strong><br/>
<a href="https://www.wiwasolvet.ca">www.wiwasolvet.ca</a></p>
<p><strong>If you found this software helpful please consider donating at:</strong><br/>
<a href="https://www.patreon.com/wiwasolvet">www.patreon.com/wiwasolvet</a></p>

<p><strong>If you would like to stay connected on social media the accounts are:</strong><br/>
Facebook: <a href="https://www.facebook.com/wiwasolvet/">www.facebook.com/wiwasolvet/</a><br/>
Twitter: <a href="http://twitter.com/wiwasolvet">twitter.com/wiwasolvet</a><br/>
LinkedIn: <a href="https://www.linkedin.com/in/jacobthompsonwiwasolvet/">www.linkedin.com/in/jacobthompsonwiwasolvet/</a></p>

<h3><a href="https://choosealicense.com/licenses/gpl-3.0/">Licence: GNU GPLv3</a></h3>