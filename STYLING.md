# Styling

The styling of this website is happening in a number of different places. some of the highlevel changes are set in the `format` section of [_quarto.yml](_quarto.yml), with the rest of the main styles set in [styles.scss](styles.scss).

The front page includes a number of detailed styling, these are all located in [styles-frontpage.scss](styles-frontpage.scss). They are all wrapped in `#FrontPage` ID so they shouldn't affect anything not located in the front page.

The sidebar for the [Get Started](start/) section has a unique style, and that is specified in the [start/styles.css](start/styles.css) file, that is loaded into each of these pages with either `css: styles.css` or `css: ../styles.css`.
