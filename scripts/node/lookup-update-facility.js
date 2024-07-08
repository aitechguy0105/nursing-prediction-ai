//////////////////////////////////////////////////////////////////////////////////////
// Utility to lookup facilities with no lon/lat and update the address info using 
// Google Maps places API. Uses node.js. 

//
// 1) Setup .env file with the following env variables:
//
// PGUSER=<Webapp DB username>
// PGHOST=<Webapp DB host>
// PGPASSWORD=<Webapp DB password>
// PGDATABASE=webapp
// PGPORT=5432
//
//
// 2) Install packages:
//    npm install axios pg dotenv parse-address --save

let axios = require('axios');
const { Pool } = require('pg')
require('dotenv').config();
var parser = require('parse-address'); 

async function geocodingQuery(facility) {
    try {
        const facilityName = encodeURIComponent(`${facility}`.replace(/ /g, '+'))
        let url = `https://maps.googleapis.com/maps/api/place/findplacefromtext/json?fields=formatted_address%2Cname%2Crating%2Copening_hours%2Cgeometry&input=${facilityName}&inputtype=textquery&key=AIzaSyDcM8dp3n8fFMzIJaCVMibvlTf5v4RK9dQ`;
        let ret = await axios.get(url);
        if ( ret.data.candidates.length > 0 ) {
            return {success: true, address: ret.data.candidates[0].formatted_address, lat: ret.data.candidates[0].geometry.location.lat, lng: ret.data.candidates[0].geometry.location.lng}
        }
        else {
            return {success: false, error: 'No result found'};
        }
    }
    catch(e) {
        console.log(e);
    }
}

async function main() {
    try {
        const pool = new Pool();
        const client = await pool.connect();
        let sql = 
`
    select 
        webapp_facility.id, concat(webapp_organization.name, ' ', webapp_facility.name, ' ', 'skilled nursing facility') as phrase
    FROM 
        webapp_facility
    INNER JOIN webapp_region
    ON webapp_region.id = webapp_facility.region_id
    INNER JOIN webapp_organization
    ON webapp_region.organization_id = webapp_organization.id
    WHERE 
        webapp_facility.status = 'active' AND webapp_facility.latitude IS NULL AND webapp_facility.longitude IS NULL
`;
        const ret = await client.query(sql);
        if ( ret.rows.length == 0 ) {
            await client.end();
            throw(new Error("Could not find any matching facilities."))
        }

        for ( let i=0; i<ret.rows.length; i++ ) {
            let geo = await geocodingQuery(ret.rows[i].phrase);
            if ( geo.success ) {
                console.log(ret.rows[i].phrase);
                console.log(JSON.stringify(geo, null, 4));
                let addressParts = parser.parseLocation(geo.address);
                console.log(JSON.stringify(addressParts, null, 4));
                await client.query(`UPDATE webapp_facility SET address1 = '${addressParts.number} ${addressParts.street} ${addressParts.type}', zipcode = '${addressParts.zip}', city = '${addressParts.city}', state = '${addressParts.state}', longitude = ${geo.lng}, latitude = ${geo.lat} where id = ${ret.rows[i].id}`);
            }
            else {
                console.log('Geo info for ' + ret.rows[i].phrase + ' not found');
            }
        }
        await client.end()
    }
    catch(e) {
        console.log(e.message);
    }
}

main();