-- FUNCTION: public.trigger_set_timestamp()

-- DROP FUNCTION public.trigger_set_timestamp();

CREATE OR REPLACE FUNCTION public.trigger_set_timestamp()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
  NEW.UpdatedAt = NOW();
  RETURN NEW;
END;
$function$
;

