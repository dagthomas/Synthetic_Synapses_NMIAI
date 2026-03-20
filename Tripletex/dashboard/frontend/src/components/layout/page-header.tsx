interface PageHeaderProps {
  title: string
  description?: string
  children?: React.ReactNode
}

export function PageHeader({ title, description, children }: PageHeaderProps) {
  return (
    <div className="flex items-start justify-between mb-5">
      <div>
        <h2 className="text-xl font-bold tracking-tight text-foreground">
          {title}
        </h2>
        {description && (
          <p className="text-[13px] text-muted-foreground mt-1 max-w-lg">
            {description}
          </p>
        )}
      </div>
      {children && (
        <div className="flex items-center gap-2 pt-0.5">{children}</div>
      )}
    </div>
  )
}
