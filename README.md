# SuperStore Analytics

## Purpose

The goal of this project was to explore a retail company's sales data, transform it into a form suitable for reporting, and present the main business insights through visualisations.

## Data

The dataset (`super_store_us.xlsx`) covers Super Store's US sales for January–June 2015 and consists of three sheets:

- **Orders:** 1,952 order lines with customer, product, location, shipping, discount, sales and profit details. In total it covers about $1.92M in sales and $224k in profit.
- **Returns:** order IDs that were returned. These are linked to Orders through `Order ID`.
- **Users:** the manager responsible for each of the four regions (Central, East, South, West). These are linked to Orders through `Region`.

Before analysis, the data was cleaned. This fixed inconsistent text values and postal codes that had lost their leading zeros, handled a duplicated row ID, and filled missing product margins. The three sheets were then joined, and new fields such as shipping time, profit margin and discount bands were added.

## Conclusions

- **High discounts remove the profit.** Lines with a 1–3% discount earn about a 21% margin, but lines with a 7%+ discount earn almost nothing.
- **Small orders lose money.** Order lines under $500 make up two thirds of all lines and lose money in total, largely because of shipping costs.
- **South is the only loss-making region.** Its Technology sales drive most of the loss.
- **Office Supplies is the most profitable category** even though it has the lowest sales. Tables and Bookcases lose money.
- **Profitability improved over the period.** Margin rose from near 0% in early 2015 to about 20% by May–June, while sales stayed roughly flat.
