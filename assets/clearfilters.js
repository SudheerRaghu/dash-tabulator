window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        clear_filters: function(filters, rows,table) {
            console.log(table)
            table.clearFilters;
            // table.clearFilter(true);
            table.clearHeaderFilter;
            console.log(filters);
            console.log("Clear filters clicked");
        }
    }
});