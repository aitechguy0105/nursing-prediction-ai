-- Drop table

-- DROP TABLE public.mds_items;

CREATE TABLE public.mds_items (
	mds_item varchar NOT NULL,
	item_value varchar NOT NULL,
	"name" varchar NULL,
	CONSTRAINT mds_items_pk PRIMARY KEY (mds_item, item_value)
);

-- Permissions

ALTER TABLE public.mds_items OWNER TO saivaadmin;
GRANT ALL ON TABLE public.mds_items TO saivaadmin;
GRANT SELECT ON TABLE public.mds_items TO quicksight_ro;
